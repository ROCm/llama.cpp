#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static ggml_context_ptr make_context() {
    ggml_init_params params = {
        /* .mem_size   = */ 256 * ggml_tensor_overhead() + ggml_graph_overhead_custom(96, false),
        /* .mem_buffer = */ nullptr,
        /* .no_alloc   = */ true,
    };
    return ggml_context_ptr(ggml_init(params));
}

static void expect_eq(const std::vector<float> & actual, const std::vector<float> & expected, const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g\n",
                label, i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void expect_near(
        const std::vector<float> & actual,
        const std::vector<float> & expected,
        float tolerance,
        const char * label) {
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (std::fabs(actual[i] - expected[i]) > tolerance) {
            std::fprintf(stderr, "%s[%zu]: got %.9g expected %.9g tolerance %.9g\n",
                label, i, actual[i], expected[i], tolerance);
            std::abort();
        }
    }
}

static std::vector<float> tensor_to_float(const ggml_tensor * tensor) {
    std::vector<uint8_t> data(ggml_nbytes(tensor));
    ggml_backend_tensor_get(tensor, data.data(), 0, data.size());

    const auto * traits = ggml_get_type_traits(tensor->type);
    const size_t block_size = ggml_blck_size(tensor->type);
    const bool quantized = ggml_is_quantized(tensor->type);
    std::vector<float> block_values(block_size);
    std::vector<float> values;
    values.reserve(ggml_nelements(tensor));

    for (int64_t i3 = 0; i3 < tensor->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < tensor->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < tensor->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < tensor->ne[0]; i0 += block_size) {
                    const size_t offset =
                        static_cast<size_t>(i3) * tensor->nb[3] +
                        static_cast<size_t>(i2) * tensor->nb[2] +
                        static_cast<size_t>(i1) * tensor->nb[1] +
                        static_cast<size_t>(i0 / block_size) * tensor->nb[0];
                    if (tensor->type == GGML_TYPE_F32) {
                        float value = 0.0f;
                        std::memcpy(&value, data.data() + offset, sizeof(value));
                        values.push_back(value);
                    } else if (tensor->type == GGML_TYPE_F16) {
                        ggml_fp16_t value = 0;
                        std::memcpy(&value, data.data() + offset, sizeof(value));
                        values.push_back(ggml_fp16_to_fp32(value));
                    } else if (quantized) {
                        traits->to_float(data.data() + offset, block_values.data(), block_size);
                        values.insert(values.end(), block_values.begin(), block_values.end());
                    } else {
                        GGML_ABORT("unsupported tensor_to_float type");
                    }
                }
            }
        }
    }
    return values;
}

static std::vector<int32_t> tensor_to_i32(const ggml_tensor * tensor) {
    GGML_ASSERT(tensor->type == GGML_TYPE_I32);
    std::vector<int32_t> values(ggml_nelements(tensor));
    ggml_backend_tensor_get(tensor, values.data(), 0, values.size() * sizeof(int32_t));
    return values;
}

static std::vector<float> reference_mul_mat(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        int64_t k,
        int64_t rows,
        int64_t cols) {
    std::vector<float> output(static_cast<size_t>(rows * cols), 0.0f);
    for (int64_t col = 0; col < cols; ++col) {
        for (int64_t row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int64_t i = 0; i < k; ++i) {
                sum += lhs[static_cast<size_t>(row * k + i)] * rhs[static_cast<size_t>(col * k + i)];
            }
            output[static_cast<size_t>(col * rows + row)] = sum;
        }
    }
    return output;
}

static std::vector<float> reference_mul_mat_batched(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        int64_t k,
        int64_t rows,
        int64_t cols,
        int64_t lhs_ne2,
        int64_t lhs_ne3,
        int64_t out_ne2,
        int64_t out_ne3) {
    std::vector<float> output(static_cast<size_t>(rows * cols * out_ne2 * out_ne3), 0.0f);
    for (int64_t i3 = 0; i3 < out_ne3; ++i3) {
        const int64_t lhs_i3 = lhs_ne3 == out_ne3 ? i3 : i3 / (out_ne3 / lhs_ne3);
        for (int64_t i2 = 0; i2 < out_ne2; ++i2) {
            const int64_t lhs_i2 = lhs_ne2 == out_ne2 ? i2 : i2 / (out_ne2 / lhs_ne2);
            for (int64_t col = 0; col < cols; ++col) {
                for (int64_t row = 0; row < rows; ++row) {
                    float sum = 0.0f;
                    for (int64_t i = 0; i < k; ++i) {
                        const size_t lhs_idx = static_cast<size_t>(
                            i + k * (row + rows * (lhs_i2 + lhs_ne2 * lhs_i3)));
                        const size_t rhs_idx = static_cast<size_t>(
                            i + k * (col + cols * (i2 + out_ne2 * i3)));
                        sum += lhs[lhs_idx] * rhs[rhs_idx];
                    }
                    output[static_cast<size_t>(row + rows * (col + cols * (i2 + out_ne2 * i3)))] = sum;
                }
            }
        }
    }
    return output;
}

static void prepare_rows(
        ggml_type type,
        int64_t ncols,
        int64_t nrows,
        const std::vector<float> & input,
        std::vector<float> & reference,
        std::vector<uint8_t> & storage) {
    reference.resize(input.size());
    if (type == GGML_TYPE_F32) {
        storage.resize(input.size() * sizeof(float));
        std::memcpy(storage.data(), input.data(), storage.size());
        reference = input;
        return;
    }

    const ggml_type_traits * traits = ggml_get_type_traits(type);
    GGML_ASSERT(traits->from_float_ref != nullptr);
    GGML_ASSERT(traits->to_float != nullptr);
    const size_t row_bytes = ggml_row_size(type, ncols);
    storage.assign(row_bytes * static_cast<size_t>(nrows), 0);
    for (int64_t row = 0; row < nrows; ++row) {
        const int64_t row_offset = row * ncols;
        uint8_t * row_data = storage.data() + row_bytes * static_cast<size_t>(row);
        traits->from_float_ref(input.data() + row_offset, row_data, ncols);
        traits->to_float(row_data, reference.data() + row_offset, ncols);
    }
}

static void prepare_mul_mat_lhs(
        ggml_type type,
        int64_t k,
        int64_t rows,
        const std::vector<float> & input,
        std::vector<float> & reference,
        std::vector<uint8_t> & storage) {
    prepare_rows(type, k, rows, input, reference, storage);
}

static void run_mul_mat_vec_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        ggml_type lhs_type,
        int64_t k,
        int64_t rows,
        int64_t cols,
        float tolerance,
        const char * label) {
    GGML_ASSERT(k % ggml_blck_size(lhs_type) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), lhs_type, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 5) % 101) - 50) / 37.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 11) % 89) - 44) / 41.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(lhs_type, k, rows, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(static_cast<size_t>(rows * cols), -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols), tolerance, label);
}

static void run_mul_mat_vec_batched_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        ggml_type lhs_type,
        int64_t k,
        int64_t rows,
        int64_t cols,
        int64_t lhs_ne2,
        int64_t lhs_ne3,
        int64_t out_ne2,
        int64_t out_ne3,
        float tolerance,
        const char * label) {
    GGML_ASSERT(k % ggml_blck_size(lhs_type) == 0);
    GGML_ASSERT(out_ne2 % lhs_ne2 == 0);
    GGML_ASSERT(out_ne3 % lhs_ne3 == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), lhs_type, k, rows, lhs_ne2, lhs_ne3);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, k, cols, out_ne2, out_ne3);
    ggml_tensor * out = ggml_mul_mat(ctx.get(), lhs, rhs);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * lhs_ne2 * lhs_ne3));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols * out_ne2 * out_ne3));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 113) - 56) / 43.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 3) % 97) - 48) / 47.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(lhs_type, k, rows * lhs_ne2 * lhs_ne3, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    expect_near(
        tensor_to_float(out),
        reference_mul_mat_batched(lhs_reference, rhs_f32, k, rows, cols, lhs_ne2, lhs_ne3, out_ne2, out_ne3),
        tolerance, label);
}

static size_t index_4d(int64_t i0, int64_t i1, int64_t i2, int64_t i3, int64_t ne0, int64_t ne1, int64_t ne2) {
    return static_cast<size_t>(i0 + ne0 * (i1 + ne1 * (i2 + ne2 * i3)));
}

static std::vector<float> reference_flash_attn_ext_decode(
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & mask,
        const std::vector<float> & sinks,
        int64_t d,
        int64_t n,
        int64_t h,
        int64_t h_kv,
        int64_t kv,
        int64_t s,
        float scale,
        bool has_mask,
        bool has_sinks) {
    std::vector<float> output(static_cast<size_t>(d * h * n * s), 0.0f);
    std::vector<float> logits(static_cast<size_t>(kv));
    for (int64_t seq = 0; seq < s; ++seq) {
        for (int64_t token = 0; token < n; ++token) {
            for (int64_t head = 0; head < h; ++head) {
                const int64_t kv_head = head / (h / h_kv);
                float max_score = has_sinks ? sinks[static_cast<size_t>(head)] : -INFINITY;
                for (int64_t t = 0; t < kv; ++t) {
                    float score = 0.0f;
                    for (int64_t col = 0; col < d; ++col) {
                        score +=
                            q[index_4d(col, token, head, seq, d, n, h)] *
                            k[index_4d(col, t, kv_head, seq, d, kv, h_kv)];
                    }
                    score *= scale;
                    if (has_mask) {
                        score += mask[index_4d(t, token, 0, seq, kv, n, 1)];
                    }
                    logits[static_cast<size_t>(t)] = score;
                    max_score = std::max(max_score, score);
                }

                float sum = has_sinks ? std::exp(sinks[static_cast<size_t>(head)] - max_score) : 0.0f;
                for (int64_t t = 0; t < kv; ++t) {
                    logits[static_cast<size_t>(t)] = std::exp(logits[static_cast<size_t>(t)] - max_score);
                    sum += logits[static_cast<size_t>(t)];
                }

                for (int64_t col = 0; col < d; ++col) {
                    float value = 0.0f;
                    for (int64_t t = 0; t < kv; ++t) {
                        value +=
                            logits[static_cast<size_t>(t)] *
                            v[index_4d(col, t, kv_head, seq, d, kv, h_kv)];
                    }
                    output[index_4d(col, head, token, seq, d, h, n)] = value / sum;
                }
            }
        }
    }
    return output;
}

static void run_flash_attn_ext_decode_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        ggml_type k_type,
        ggml_type v_type,
        bool mask_enabled,
        bool sinks_enabled,
        const char * label) {
    const int64_t d = 32;
    const int64_t n = 3;
    const int64_t h = 4;
    const int64_t h_kv = 2;
    const int64_t kv = 5;
    const int64_t s = 2;
    GGML_ASSERT(d % ggml_blck_size(k_type) == 0);
    GGML_ASSERT(d % ggml_blck_size(v_type) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, d, n, h, s);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), k_type, d, kv, h_kv, s);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), v_type, d, kv, h_kv, s);
    ggml_tensor * mask = mask_enabled ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, kv, n, 1, s) : nullptr;
    ggml_tensor * sinks = sinks_enabled ? ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, h) : nullptr;
    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, 1.0f / std::sqrt(static_cast<float>(d)), 0.0f, 0.0f);
    ggml_flash_attn_ext_add_sinks(out, sinks);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(d * n * h * s));
    std::vector<float> k_data(static_cast<size_t>(d * kv * h_kv * s));
    std::vector<float> v_data(static_cast<size_t>(d * kv * h_kv * s));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 41) - 20) / 29.0f;
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 37) - 18) / 31.0f;
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = static_cast<float>(static_cast<int>((i * 13 + 9) % 43) - 21) / 23.0f;
    }

    std::vector<float> k_reference;
    std::vector<float> v_reference;
    std::vector<uint8_t> k_storage;
    std::vector<uint8_t> v_storage;
    prepare_rows(k_type, d, kv * h_kv * s, k_data, k_reference, k_storage);
    prepare_rows(v_type, d, kv * h_kv * s, v_data, v_reference, v_storage);

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_storage.data(), 0, k_storage.size());
    ggml_backend_tensor_set(v, v_storage.data(), 0, v_storage.size());

    std::vector<float> mask_reference;
    std::vector<ggml_fp16_t> mask_storage;
    if (mask_enabled) {
        mask_reference.resize(static_cast<size_t>(kv * n * s));
        mask_storage.resize(mask_reference.size());
        for (int64_t seq = 0; seq < s; ++seq) {
            for (int64_t token = 0; token < n; ++token) {
                for (int64_t t = 0; t < kv; ++t) {
                    const float value = t > token + 1 ? -1000.0f : 0.125f * static_cast<float>(token - t);
                    mask_reference[index_4d(t, token, 0, seq, kv, n, 1)] = value;
                    mask_storage[index_4d(t, token, 0, seq, kv, n, 1)] = ggml_fp32_to_fp16(value);
                }
            }
        }
        ggml_backend_tensor_set(mask, mask_storage.data(), 0, mask_storage.size() * sizeof(ggml_fp16_t));
    }

    std::vector<float> sink_data;
    if (sinks_enabled) {
        sink_data.resize(static_cast<size_t>(h));
        for (int64_t head = 0; head < h; ++head) {
            sink_data[static_cast<size_t>(head)] = -0.5f + 0.25f * static_cast<float>(head);
        }
        ggml_backend_tensor_set(sinks, sink_data.data(), 0, sink_data.size() * sizeof(float));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(
        tensor_to_float(out),
        reference_flash_attn_ext_decode(
            q_data, k_reference, v_reference, mask_reference, sink_data,
            d, n, h, h_kv, kv, s,
            1.0f / std::sqrt(static_cast<float>(d)),
            mask_enabled,
            sinks_enabled),
        5.0e-3f,
        label);
}

static void run_add_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * sum = ggml_add(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) - 8.0f;
        rhs_data[i] = static_cast<float>(i % 11) * 0.5f;
        expected[i] = lhs_data[i] + rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "ADD graph failed for n=%" PRId64 ": %s\n", n, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(sum, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "add");
}

static void run_mul_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * product = ggml_mul(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, product);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) - 8.0f;
        rhs_data[i] = static_cast<float>(i % 11) * 0.25f;
        expected[i] = lhs_data[i] * rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "MUL graph failed for n=%" PRId64 ": %s\n", n, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(product, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "mul");
}

static void run_broadcast_case(ggml_backend_t backend, enum ggml_op op) {
    static constexpr int64_t ne0 = 257;
    static constexpr int64_t ne1 = 3;
    static constexpr int64_t ne2 = 2;
    static constexpr int64_t ne3 = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2, ne3);
    ggml_tensor * rhs = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, ne1, 1, ne3);
    ggml_tensor * out = nullptr;
    if (op == GGML_OP_ADD) {
        out = ggml_add(ctx.get(), lhs, rhs);
    } else if (op == GGML_OP_MUL) {
        out = ggml_mul(ctx.get(), lhs, rhs);
    } else {
        out = ggml_div(ctx.get(), lhs, rhs);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t lhs_count = ne0 * ne1 * ne2 * ne3;
    const int64_t rhs_count = ne1 * ne3;
    std::vector<float> lhs_data(lhs_count);
    std::vector<float> rhs_data(rhs_count);
    std::vector<float> expected(lhs_count);
    for (int64_t i = 0; i < lhs_count; ++i) {
        lhs_data[i] = static_cast<float>(i % 19) - 9.0f;
    }
    for (int64_t i = 0; i < rhs_count; ++i) {
        rhs_data[i] = static_cast<float>(i + 2);
    }
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                for (int64_t i0 = 0; i0 < ne0; ++i0) {
                    const int64_t lhs_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;
                    const float a = lhs_data[lhs_idx];
                    const float b = rhs_data[i3 * ne1 + i1];
                    expected[lhs_idx] = op == GGML_OP_ADD ? a + b : (op == GGML_OP_MUL ? a * b : a / b);
                }
            }
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "broadcast graph failed for op=%d: %s\n", op, ggml_status_to_string(status));
        std::abort();
    }

    std::vector<float> actual(lhs_count, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "broadcast");
}

static void run_scale_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_scale_bias(ctx.get(), src, 1.25f, -0.75f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 13) - 6.0f;
        expected[i] = src_data[i] * 1.25f - 0.75f;
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-6f, "scale");
}

static void run_unary_case(ggml_backend_t backend, enum ggml_unary_op op, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_unary(ctx.get(), src, op);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
        const float x = src_data[i];
        if (op == GGML_UNARY_OP_SILU) {
            expected[i] = x / (1.0f + std::exp(-x));
        } else if (op == GGML_UNARY_OP_SIGMOID) {
            expected[i] = 1.0f / (1.0f + std::exp(-x));
        } else {
            expected[i] = std::log1p(std::exp(x));
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-4f, ggml_unary_op_name(op));
}

static void run_swiglu_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * rhs = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), lhs, rhs);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(n);
    std::vector<float> rhs_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        lhs_data[i] = static_cast<float>(i % 17) * 0.25f - 2.0f;
        rhs_data[i] = static_cast<float>(i % 7) - 3.0f;
        expected[i] = lhs_data[i] / (1.0f + std::exp(-lhs_data[i])) * rhs_data[i];
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_near(actual, expected, 1e-4f, "swiglu");
}

static void run_clamp_case(ggml_backend_t backend, int64_t n) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, n);
    ggml_tensor * out = ggml_clamp(ctx.get(), src, -1.5f, 2.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(n);
    std::vector<float> expected(n);
    for (int64_t i = 0; i < n; ++i) {
        src_data[i] = static_cast<float>(i % 17) - 8.0f;
        expected[i] = src_data[i] < -1.5f ? -1.5f : (src_data[i] > 2.0f ? 2.0f : src_data[i]);
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> actual(n, -1.0f);
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size() * sizeof(float));
    expect_eq(actual, expected, "clamp");
}

static void run_cpy_strided_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 6, 4);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, 3, 4, base->nb[1], sizeof(float));
    ggml_tensor * target = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 4);
    ggml_tensor * out = ggml_cpy(ctx.get(), view, target);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(24);
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(i);
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int row = 0; row < 4; ++row) {
        for (int col = 1; col < 4; ++col) {
            expected.push_back(base_data[row * 6 + col]);
        }
    }
    expect_eq(tensor_to_float(out), expected, "cpy_strided");
}

static void run_cpy_f32_f16_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 257);
    ggml_tensor * dst = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F16, 257);
    ggml_tensor * out = ggml_cpy(ctx.get(), src, dst);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(257);
    std::vector<float> expected(257);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>(i % 19) - 9) * 0.125f;
        expected[i] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(src_data[i]));
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_eq(tensor_to_float(out), expected, "cpy_f32_f16");
}

static void run_cont_slice_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * base = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 7, 3);
    ggml_tensor * view = ggml_view_2d(ctx.get(), base, 4, 3, base->nb[1], 2 * sizeof(float));
    ggml_tensor * out = ggml_cont(ctx.get(), view);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> base_data(21);
    for (size_t i = 0; i < base_data.size(); ++i) {
        base_data[i] = static_cast<float>(100 + i);
    }
    ggml_backend_tensor_set(base, base_data.data(), 0, base_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int row = 0; row < 3; ++row) {
        for (int col = 2; col < 6; ++col) {
            expected.push_back(base_data[row * 7 + col]);
        }
    }
    expect_eq(tensor_to_float(out), expected, "cont_slice");
}

static void run_get_rows_case(ggml_backend_t backend, int64_t rows_to_get) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 5, 6, 2);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows_to_get, 2);
    ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(5 * 6 * 2);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(i);
    }
    std::vector<int32_t> row_data(rows_to_get * 2);
    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            row_data[batch * rows_to_get + row] = static_cast<int32_t>((row * 2 + batch) % 6);
        }
    }

    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    for (int64_t batch = 0; batch < 2; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            const int64_t selected = row_data[batch * rows_to_get + row];
            for (int64_t col = 0; col < 5; ++col) {
                expected.push_back(src_data[batch * 30 + selected * 5 + col]);
            }
        }
    }
    expect_eq(tensor_to_float(out), expected, rows_to_get == 1 ? "get_rows_nr1" : "get_rows");
}

static void run_get_rows_q5_k_case(
        ggml_backend_t backend,
        int64_t ncols,
        int64_t src_rows,
        int64_t rows_to_get,
        int64_t batches) {
    const int64_t block_size = ggml_blck_size(GGML_TYPE_Q5_K);
    GGML_ASSERT(ncols % block_size == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q5_K, ncols, src_rows, batches);
    ggml_tensor * rows = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, rows_to_get, batches);
    ggml_tensor * out = ggml_get_rows(ctx.get(), src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const size_t row_bytes = ggml_row_size(GGML_TYPE_Q5_K, ncols);
    const ggml_type_traits * traits = ggml_get_type_traits(GGML_TYPE_Q5_K);
    GGML_ASSERT(traits->from_float_ref != nullptr);
    GGML_ASSERT(traits->to_float != nullptr);
    std::vector<float> src_f32(static_cast<size_t>(ncols * src_rows * batches));
    std::vector<float> src_dequant(src_f32.size());
    std::vector<uint8_t> src_q5(row_bytes * static_cast<size_t>(src_rows * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < src_rows; ++row) {
            const int64_t row_offset = (batch * src_rows + row) * ncols;
            for (int64_t col = 0; col < ncols; ++col) {
                const int64_t i = row_offset + col;
                src_f32[static_cast<size_t>(i)] =
                    static_cast<float>(static_cast<int>((i * 17 + batch * 13 + row * 7) % 97) - 48) / 19.0f;
            }
            uint8_t * q5_row = src_q5.data() + row_bytes * static_cast<size_t>(batch * src_rows + row);
            traits->from_float_ref(src_f32.data() + row_offset, q5_row, ncols);
            traits->to_float(q5_row, src_dequant.data() + row_offset, ncols);
        }
    }

    std::vector<int32_t> row_data(static_cast<size_t>(rows_to_get * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            row_data[static_cast<size_t>(batch * rows_to_get + row)] =
                static_cast<int32_t>((row * 3 + batch + 1) % src_rows);
        }
    }

    ggml_backend_tensor_set(src, src_q5.data(), 0, src_q5.size());
    ggml_backend_tensor_set(rows, row_data.data(), 0, row_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected;
    expected.reserve(static_cast<size_t>(ncols * rows_to_get * batches));
    for (int64_t batch = 0; batch < batches; ++batch) {
        for (int64_t row = 0; row < rows_to_get; ++row) {
            const int64_t selected = row_data[static_cast<size_t>(batch * rows_to_get + row)];
            const int64_t row_offset = (batch * src_rows + selected) * ncols;
            expected.insert(
                expected.end(),
                src_dequant.begin() + row_offset,
                src_dequant.begin() + row_offset + ncols);
        }
    }
    expect_eq(tensor_to_float(out), expected, "get_rows_q5_k");
}

static void run_concat_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 3, 2);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 2, 2);
    ggml_tensor * out = ggml_concat(ctx.get(), lhs, rhs, 0);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const std::vector<float> lhs_data = { 1, 2, 3, 4, 5, 6 };
    const std::vector<float> rhs_data = { 10, 11, 12, 13 };
    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(rhs, rhs_data.data(), 0, rhs_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_eq(tensor_to_float(out), { 1, 2, 3, 10, 11, 4, 5, 6, 12, 13 }, "concat");
}

static void run_set_rows_case(ggml_backend_t backend, ggml_type type) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), type, 32, 4);
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 2);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 2);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    std::vector<float> src_data(64);
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>(static_cast<int>(i % 17) - 8) * 0.125f;
    }
    const int64_t row_data[2] = { 1, 3 };

    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    const std::vector<float> actual = tensor_to_float(out);
    const float tolerance = type == GGML_TYPE_Q4_0 ? 0.15f : (type == GGML_TYPE_Q8_0 ? 0.01f : 0.0f);
    for (int row = 0; row < 4; ++row) {
        const int src_row = row == 1 ? 0 : (row == 3 ? 1 : -1);
        for (int col = 0; col < 32; ++col) {
            const float expected = src_row >= 0 ? src_data[src_row * 32 + col] : 0.0f;
            const float got = actual[row * 32 + col];
            if (std::fabs(got - expected) > tolerance) {
                std::fprintf(stderr, "set_rows(%s)[%d,%d]: got %.9g expected %.9g tolerance %.9g\n",
                    ggml_type_name(type), row, col, got, expected, tolerance);
                std::abort();
            }
        }
    }
}

struct test_block_q8_0 {
    ggml_fp16_t d;
    int8_t qs[32];
};

static_assert(sizeof(test_block_q8_0) == sizeof(ggml_fp16_t) + 32, "unexpected q8_0 test block size");

static void run_set_rows_q8_0_tie_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, 32, 2);
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 32, 1);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, 1);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, src, rows);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    std::vector<float> src_data(32, 0.0f);
    src_data[0] = 127.0f;
    src_data[1] = 0.5f;
    src_data[2] = -0.5f;
    src_data[3] = 1.5f;
    src_data[4] = -1.5f;
    src_data[5] = 2.5f;
    src_data[6] = -2.5f;
    const int64_t row_data[1] = { 1 };

    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(rows, row_data, 0, sizeof(row_data));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<uint8_t> actual(ggml_nbytes(out));
    ggml_backend_tensor_get(out, actual.data(), 0, actual.size());
    const test_block_q8_0 * blocks = reinterpret_cast<const test_block_q8_0 *>(actual.data());
    const int8_t expected[] = { 127, 1, -1, 2, -2, 3, -3 };
    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); ++i) {
        if (blocks[1].qs[i] != expected[i]) {
            std::fprintf(stderr, "set_rows(q8_0 tie) qs[%zu]: got %d expected %d\n",
                i, static_cast<int>(blocks[1].qs[i]), static_cast<int>(expected[i]));
            std::abort();
        }
    }
}

static std::vector<float> rowwise_input(int64_t ncols, int64_t nrows) {
    std::vector<float> data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < data.size(); ++i) {
        const int value = static_cast<int>((i * 17 + 11) % 29) - 14;
        data[i] = static_cast<float>(value) * 0.0625f;
    }
    return data;
}

static void run_rms_norm_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    constexpr float eps = 1.0e-6f;
    ggml_tensor * out = ggml_rms_norm(ctx.get(), src, eps);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[row * ncols + col];
            sum += value * value;
        }
        const float scale = 1.0f / std::sqrt(sum / static_cast<float>(ncols) + eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] = src_data[row * ncols + col] * scale;
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "rms_norm");
}

static void run_sum_rows_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    ggml_tensor * out = ggml_sum_rows(ctx.get(), src);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            sum += src_data[row * ncols + col];
        }
        expected[row] = sum;
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "sum_rows");
}

static void run_l2_norm_case(ggml_backend_t backend, int64_t ncols, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2);
    constexpr float eps = 1.0e-7f;
    ggml_tensor * out = ggml_l2_norm(ctx.get(), src, eps);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    const std::vector<float> src_data = rowwise_input(ncols, nrows);
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float value = src_data[row * ncols + col];
            sum += value * value;
        }
        const float denom = std::sqrt(sum);
        const float scale = 1.0f / (denom > eps ? denom : eps);
        for (int64_t col = 0; col < ncols; ++col) {
            expected[row * ncols + col] = src_data[row * ncols + col] * scale;
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "l2_norm");
}

static void run_soft_max_case(ggml_backend_t backend, int64_t ncols, bool mask) {
    static constexpr int64_t ne1 = 3;
    static constexpr int64_t ne2 = 2;
    static constexpr int64_t ne3 = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2, ne3);
    ggml_tensor * mask_tensor = mask ? ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, ncols, ne1, ne2, ne3) : nullptr;
    constexpr float scale = 0.25f;
    ggml_tensor * out = ggml_soft_max_ext(ctx.get(), src, mask_tensor, scale, 0.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2 * ne3;
    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] = static_cast<float>((row * 17 + col * 7) % 31) * 0.1f - 1.5f;
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));

    std::vector<float> mask_data;
    if (mask_tensor) {
        mask_data.resize(static_cast<size_t>(ncols * ne1 * ne2 * ne3));
        for (int64_t row = 0; row < ne1 * ne2 * ne3; ++row) {
            for (int64_t col = 0; col < ncols; ++col) {
                mask_data[row * ncols + col] = static_cast<float>((row + col) % 5) * -0.125f;
            }
        }
        ggml_backend_tensor_set(mask_tensor, mask_data.data(), 0, mask_data.size() * sizeof(float));
    }

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(src_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        const int64_t i1 = row % ne1;
        const int64_t i2 = (row / ne1) % ne2;
        float max_value = -INFINITY;
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            max_value = std::max(max_value, src_data[row * ncols + col] * scale + bias);
        }
        float sum = 0.0f;
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            sum += std::exp(src_data[row * ncols + col] * scale + bias - max_value);
        }
        for (int64_t col = 0; col < ncols; ++col) {
            const float bias = mask_tensor ? mask_data[(i2 * ne1 + i1) * ncols + col] : 0.0f;
            expected[row * ncols + col] =
                std::exp(src_data[row * ncols + col] * scale + bias - max_value) / sum;
        }
    }
    expect_near(tensor_to_float(out), expected, 3.0e-6f, "soft_max");
}

static void run_argsort_case(ggml_backend_t backend, int64_t ncols, int64_t nrows, ggml_sort_order order) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_argsort(ctx.get(), src, order);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> src_data(static_cast<size_t>(ncols * nrows));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            src_data[row * ncols + col] =
                static_cast<float>((col * 37 + row * 11) % 257) + static_cast<float>(col) * 0.001f;
        }
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> expected(static_cast<size_t>(ncols * nrows));
    std::vector<int32_t> indices(static_cast<size_t>(ncols));
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            indices[col] = static_cast<int32_t>(col);
        }
        std::sort(indices.begin(), indices.end(), [&](int32_t a, int32_t b) {
            const float av = src_data[row * ncols + a];
            const float bv = src_data[row * ncols + b];
            return order == GGML_SORT_ORDER_ASC ? av < bv : av > bv;
        });
        std::copy(indices.begin(), indices.end(), expected.begin() + row * ncols);
    }
    const std::vector<int32_t> actual = tensor_to_i32(out);
    GGML_ASSERT(actual.size() == expected.size());
    for (size_t i = 0; i < actual.size(); ++i) {
        if (actual[i] != expected[i]) {
            std::fprintf(stderr, "argsort[%zu]: got %" PRId32 " expected %" PRId32 "\n", i, actual[i], expected[i]);
            std::abort();
        }
    }
}

static void run_rope_imrope_case(ggml_backend_t backend, int64_t ne0, int64_t ne1, int64_t ne2) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, ne0, ne1, ne2);
    ggml_tensor * pos = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, ne2 * 4);
    int sections[GGML_MROPE_SECTIONS] = {
        static_cast<int32_t>(ne0 / 4),
        static_cast<int32_t>(ne0 / 4),
        static_cast<int32_t>(ne0 / 4),
        static_cast<int32_t>(ne0 - 3 * (ne0 / 4)),
    };
    constexpr float freq_base = 10000.0f;
    constexpr float freq_scale = 1.0f;
    constexpr float attn_factor = 1.0f;
    ggml_tensor * out = ggml_rope_multi(
        ctx.get(), src, pos, nullptr, static_cast<int>(ne0), sections, GGML_ROPE_TYPE_IMROPE, 0,
        freq_base, freq_scale, 0.0f, attn_factor, 1.0f, 1.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t nrows = ne1 * ne2;
    std::vector<float> src_data(static_cast<size_t>(ne0 * nrows));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>((i * 13) % 29) * 0.05f - 0.7f;
    }
    std::vector<int32_t> pos_data(static_cast<size_t>(ne2 * 4));
    for (int64_t i = 0; i < ne2 * 4; ++i) {
        pos_data[i] = static_cast<int32_t>((i * 3) % 17);
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(pos, pos_data.data(), 0, pos_data.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected = src_data;
    const int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    const float theta_scale = std::pow(freq_base, -2.0f / static_cast<float>(ne0));
    for (int64_t i2 = 0; i2 < ne2; ++i2) {
        for (int64_t i1 = 0; i1 < ne1; ++i1) {
            const int64_t row_base = (i2 * ne1 + i1) * ne0;
            for (int64_t pair = 0; pair < ne0 / 2; ++pair) {
                const int32_t i0 = static_cast<int32_t>(2 * pair);
                if (i0 >= ne0) {
                    continue;
                }
                const int32_t sector = (i0 / 2) % sect_dims;
                const int32_t pos_idx =
                    (sector % 3 == 1 && sector < 3 * sections[1]) ? 1 :
                    (sector % 3 == 2 && sector < 3 * sections[2]) ? 2 :
                    (sector % 3 == 0 && sector < 3 * sections[0]) ? 0 : 3;
                const float theta = static_cast<float>(pos_data[i2 + ne2 * pos_idx]) *
                    std::pow(theta_scale, static_cast<float>(i0) / 2.0f) * freq_scale;
                const float cos_theta = std::cos(theta) * attn_factor;
                const float sin_theta = std::sin(theta) * attn_factor;
                const int64_t off0 = i0 / 2;
                const int64_t off1 = off0 + ne0 / 2;
                const float x0 = src_data[row_base + off0];
                const float x1 = src_data[row_base + off1];
                expected[row_base + off0] = x0 * cos_theta - x1 * sin_theta;
                expected[row_base + off1] = x0 * sin_theta + x1 * cos_theta;
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "rope_imrope");
}

static void run_ssm_conv_case(ggml_backend_t backend, int64_t d_conv, int64_t d_inner, int64_t n_tokens, int64_t n_seqs) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * src = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, d_conv - 1 + n_tokens, d_inner, n_seqs);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * out = ggml_ssm_conv(ctx.get(), src, weight);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const int64_t conv_width = d_conv - 1 + n_tokens;
    std::vector<float> src_data(static_cast<size_t>(conv_width * d_inner * n_seqs));
    std::vector<float> weight_data(static_cast<size_t>(d_conv * d_inner));
    for (size_t i = 0; i < src_data.size(); ++i) {
        src_data[i] = static_cast<float>((i * 5) % 23) * 0.1f - 1.0f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 7) % 19) * 0.05f - 0.4f;
    }
    ggml_backend_tensor_set(src, src_data.data(), 0, src_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(d_inner * n_tokens * n_seqs));
    for (int64_t seq = 0; seq < n_seqs; ++seq) {
        for (int64_t token = 0; token < n_tokens; ++token) {
            for (int64_t channel = 0; channel < d_inner; ++channel) {
                float sum = 0.0f;
                for (int64_t i = 0; i < d_conv; ++i) {
                    const float x = src_data[(seq * d_inner + channel) * conv_width + token + i];
                    const float w = weight_data[channel * d_conv + i];
                    sum += x * w;
                }
                expected[(seq * n_tokens + token) * d_inner + channel] = sum;
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 2.0e-5f, "ssm_conv");
}

static void run_gated_delta_net_case(ggml_backend_t backend, bool kda) {
    static constexpr int64_t S = 4;
    static constexpr int64_t H = 2;
    static constexpr int64_t T = 2;
    static constexpr int64_t B = 1;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, T, B);
    ggml_tensor * g = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, kda ? S : 1, H, T, B);
    ggml_tensor * beta = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, T, B);
    ggml_tensor * state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, S * S * H, B);
    ggml_tensor * out = ggml_gated_delta_net(ctx.get(), q, k, v, g, beta, state);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    const auto fill = [](std::vector<float> & values, int mul, float scale, float bias) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = static_cast<float>((i * mul) % 17) * scale + bias;
        }
    };
    std::vector<float> q_data(static_cast<size_t>(S * H * T * B));
    std::vector<float> k_data(q_data.size());
    std::vector<float> v_data(q_data.size());
    std::vector<float> g_data(static_cast<size_t>((kda ? S : 1) * H * T * B));
    std::vector<float> beta_data(static_cast<size_t>(H * T * B));
    std::vector<float> state_data(static_cast<size_t>(S * S * H * B));
    fill(q_data, 3, 0.025f, -0.2f);
    fill(k_data, 5, 0.02f, -0.15f);
    fill(v_data, 7, 0.03f, -0.1f);
    fill(g_data, 11, 0.01f, -0.05f);
    fill(beta_data, 13, 0.015f, 0.25f);
    fill(state_data, 2, 0.02f, -0.3f);
    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_data.data(), 0, k_data.size() * sizeof(float));
    ggml_backend_tensor_set(v, v_data.data(), 0, v_data.size() * sizeof(float));
    ggml_backend_tensor_set(g, g_data.data(), 0, g_data.size() * sizeof(float));
    ggml_backend_tensor_set(beta, beta_data.data(), 0, beta_data.size() * sizeof(float));
    ggml_backend_tensor_set(state, state_data.data(), 0, state_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(static_cast<size_t>(S * H * T * B + S * S * H * B));
    const float scale = 1.0f / std::sqrt(static_cast<float>(S));
    for (int64_t seq = 0; seq < B; ++seq) {
        for (int64_t head = 0; head < H; ++head) {
            for (int64_t col = 0; col < S; ++col) {
                float s_col[S];
                for (int64_t row = 0; row < S; ++row) {
                    s_col[row] = state_data[(seq * H + head) * S * S + col * S + row];
                }
                for (int64_t token = 0; token < T; ++token) {
                    float kv_col = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        const float g_row = kda ?
                            std::exp(g_data[((seq * T + token) * H + head) * S + row]) : 1.0f;
                        kv_col += g_row * s_col[row] * k_data[((seq * T + token) * H + head) * S + row];
                    }
                    const float beta_val = beta_data[(seq * T + token) * H + head];
                    const float v_col = v_data[((seq * T + token) * H + head) * S + col];
                    const float g_scalar = kda ? 1.0f : std::exp(g_data[(seq * T + token) * H + head]);
                    const float delta_col = (v_col - (kda ? kv_col : g_scalar * kv_col)) * beta_val;
                    float attn = 0.0f;
                    for (int64_t row = 0; row < S; ++row) {
                        const float g_row = kda ?
                            std::exp(g_data[((seq * T + token) * H + head) * S + row]) : g_scalar;
                        s_col[row] = g_row * s_col[row] +
                            k_data[((seq * T + token) * H + head) * S + row] * delta_col;
                        attn += s_col[row] * q_data[((seq * T + token) * H + head) * S + row];
                    }
                    expected[((seq * T + token) * H + head) * S + col] = attn * scale;
                }
                const int64_t state_offset = S * H * T * B;
                for (int64_t row = 0; row < S; ++row) {
                    expected[state_offset + (seq * H + head) * S * S + col * S + row] = s_col[row];
                }
            }
        }
    }
    expect_near(tensor_to_float(out), expected, 3.0e-5f, kda ? "gated_delta_net_kda" : "gated_delta_net");
}

static bool env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

} // namespace

int main() {
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("HRX0");
    if (!dev) {
        std::fprintf(stderr, "HRX0 not available; skipping test-backend-hrx\n");
        return 0;
    }

    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    GGML_ASSERT(buft != nullptr);
    {
        ggml_context_ptr standalone_ctx = make_context();
        ggml_tensor * standalone = ggml_new_tensor_1d(standalone_ctx.get(), GGML_TYPE_F32, 4);
        ggml_backend_buffer_ptr standalone_buffer(ggml_backend_alloc_ctx_tensors_from_buft(standalone_ctx.get(), buft));
        GGML_ASSERT(standalone_buffer != nullptr);

        const std::vector<float> standalone_input = { 10.0f, 11.0f, 12.0f, 13.0f };
        ggml_backend_tensor_set(standalone, standalone_input.data(), 0, standalone_input.size() * sizeof(float));

        std::vector<float> standalone_output(standalone_input.size(), -1.0f);
        ggml_backend_tensor_get(standalone, standalone_output.data(), 0, standalone_output.size() * sizeof(float));
        expect_eq(standalone_output, standalone_input, "standalone_output");
    }

    ggml_backend_ptr backend(ggml_backend_dev_init(dev, nullptr));
    GGML_ASSERT(backend != nullptr);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * src  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 8);
    ggml_tensor * view = ggml_view_1d(ctx.get(), src, 4, 2 * sizeof(float));
    ggml_tensor * dst  = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 4);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend.get()));
    GGML_ASSERT(buffer != nullptr);
    GGML_ASSERT(src->buffer == buffer.get());
    GGML_ASSERT(view->buffer == buffer.get());
    GGML_ASSERT(dst->buffer == buffer.get());

    const std::vector<float> input = { 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f };
    ggml_backend_tensor_set(src, input.data(), 0, input.size() * sizeof(float));

    std::vector<float> view_data(4, -1.0f);
    ggml_backend_tensor_get(view, view_data.data(), 0, view_data.size() * sizeof(float));
    expect_eq(view_data, { 2.0f, 3.0f, 4.0f, 5.0f }, "view_data");

    const std::vector<float> replacement = { 20.0f, 21.0f, 22.0f, 23.0f };
    ggml_backend_tensor_set(view, replacement.data(), 0, replacement.size() * sizeof(float));

    std::vector<float> src_after_view_set(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_view_set.data(), 0, src_after_view_set.size() * sizeof(float));
    expect_eq(src_after_view_set, { 0.0f, 1.0f, 20.0f, 21.0f, 22.0f, 23.0f, 6.0f, 7.0f }, "src_after_view_set");

    ggml_backend_tensor_copy(view, dst);
    std::vector<float> dst_data(4, -1.0f);
    ggml_backend_tensor_get(dst, dst_data.data(), 0, dst_data.size() * sizeof(float));
    expect_eq(dst_data, replacement, "dst_data");

    ggml_backend_tensor_memset(view, 0, 0, ggml_nbytes(view));
    std::vector<float> src_after_memset(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_memset.data(), 0, src_after_memset.size() * sizeof(float));
    expect_eq(src_after_memset, { 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 6.0f, 7.0f }, "src_after_memset");

    ggml_backend_buffer_clear(buffer.get(), 0);
    std::vector<float> src_after_clear(8, -1.0f);
    ggml_backend_tensor_get(src, src_after_clear.data(), 0, src_after_clear.size() * sizeof(float));
    expect_eq(src_after_clear, std::vector<float>(8, 0.0f), "src_after_clear");

    ggml_context_ptr graph_ctx = make_context();
    ggml_tensor * graph_base      = ggml_new_tensor_2d(graph_ctx.get(), GGML_TYPE_F32, 4, 3);
    ggml_tensor * graph_view      = ggml_view_1d(graph_ctx.get(), graph_base, 4, 4 * sizeof(float));
    ggml_tensor * graph_reshape   = ggml_reshape_3d(graph_ctx.get(), graph_base, 2, 2, 3);
    ggml_tensor * graph_permute   = ggml_permute(graph_ctx.get(), graph_reshape, 1, 0, 2, 3);
    ggml_tensor * graph_transpose = ggml_transpose(graph_ctx.get(), graph_base);

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx.get(), 16, false);
    ggml_build_forward_expand(graph, graph_view);
    ggml_build_forward_expand(graph, graph_reshape);
    ggml_build_forward_expand(graph, graph_permute);
    ggml_build_forward_expand(graph, graph_transpose);

    ggml_backend_buffer_ptr graph_buffer(ggml_backend_alloc_ctx_tensors(graph_ctx.get(), backend.get()));
    GGML_ASSERT(graph_buffer != nullptr);

    const auto * graph_base_data = static_cast<const uint8_t *>(graph_base->data);
    GGML_ASSERT(graph_view->data == graph_base_data + 4 * sizeof(float));
    GGML_ASSERT(graph_reshape->data == graph_base->data);
    GGML_ASSERT(graph_permute->data == graph_base->data);
    GGML_ASSERT(graph_transpose->data == graph_base->data);

    const std::vector<float> graph_input = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f,
    };
    ggml_backend_tensor_set(graph_base, graph_input.data(), 0, graph_input.size() * sizeof(float));

    const ggml_status graph_status = ggml_backend_graph_compute(backend.get(), graph);
    if (graph_status != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "metadata graph failed: %s\n", ggml_status_to_string(graph_status));
        std::abort();
    }

    std::vector<float> graph_base_after(graph_input.size(), -1.0f);
    ggml_backend_tensor_get(graph_base, graph_base_after.data(), 0, graph_base_after.size() * sizeof(float));
    expect_eq(graph_base_after, graph_input, "graph_base_after");

    std::vector<float> graph_view_data(4, -1.0f);
    ggml_backend_tensor_get(graph_view, graph_view_data.data(), 0, graph_view_data.size() * sizeof(float));
    expect_eq(graph_view_data, { 4.0f, 5.0f, 6.0f, 7.0f }, "graph_view_data");

    std::vector<float> graph_reshape_data(graph_input.size(), -1.0f);
    ggml_backend_tensor_get(graph_reshape, graph_reshape_data.data(), 0, graph_reshape_data.size() * sizeof(float));
    expect_eq(graph_reshape_data, graph_input, "graph_reshape_data");

    run_add_case(backend.get(), 1);
    run_add_case(backend.get(), 255);
    run_add_case(backend.get(), 256);
    run_add_case(backend.get(), 257);
    run_add_case(backend.get(), 1025);
    run_mul_case(backend.get(), 1);
    run_mul_case(backend.get(), 255);
    run_mul_case(backend.get(), 256);
    run_mul_case(backend.get(), 257);
    run_mul_case(backend.get(), 1025);
    run_broadcast_case(backend.get(), GGML_OP_ADD);
    run_broadcast_case(backend.get(), GGML_OP_MUL);
    run_broadcast_case(backend.get(), GGML_OP_DIV);
    run_scale_case(backend.get(), 1);
    run_scale_case(backend.get(), 256);
    run_scale_case(backend.get(), 257);
    run_cpy_strided_case(backend.get());
    run_cpy_f32_f16_case(backend.get());
    run_cont_slice_case(backend.get());
    run_set_rows_case(backend.get(), GGML_TYPE_F32);
    run_set_rows_case(backend.get(), GGML_TYPE_F16);
    run_set_rows_case(backend.get(), GGML_TYPE_Q8_0);
    run_set_rows_q8_0_tie_case(backend.get());
    run_set_rows_case(backend.get(), GGML_TYPE_Q4_0);
    run_get_rows_case(backend.get(), 1);
    run_get_rows_case(backend.get(), 3);
    run_get_rows_q5_k_case(backend.get(), ggml_blck_size(GGML_TYPE_Q5_K), 4, 1, 2);
    run_get_rows_q5_k_case(backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q5_K), 5, 3, 2);
    run_concat_case(backend.get());
    run_argsort_case(backend.get(), 1, 3, GGML_SORT_ORDER_ASC);
    run_argsort_case(backend.get(), 255, 2, GGML_SORT_ORDER_ASC);
    run_argsort_case(backend.get(), 256, 2, GGML_SORT_ORDER_DESC);
    run_ssm_conv_case(backend.get(), 1, 3, 4, 2);
    run_ssm_conv_case(backend.get(), 4, 33, 17, 2);
    if (!env_enabled("GGML_HRX_DISABLE_FAST_APPROX_PROMPT")) {
        run_rms_norm_case(backend.get(), 1, 3, 2);
        run_rms_norm_case(backend.get(), 127, 3, 2);
        run_rms_norm_case(backend.get(), 128, 3, 2);
        run_rms_norm_case(backend.get(), 129, 3, 2);
        run_rms_norm_case(backend.get(), 513, 2, 2);
        run_sum_rows_case(backend.get(), 1, 3, 2);
        run_sum_rows_case(backend.get(), 255, 3, 2);
        run_sum_rows_case(backend.get(), 256, 3, 2);
        run_sum_rows_case(backend.get(), 257, 3, 2);
        run_l2_norm_case(backend.get(), 1, 3, 2);
        run_l2_norm_case(backend.get(), 127, 3, 2);
        run_l2_norm_case(backend.get(), 128, 3, 2);
        run_l2_norm_case(backend.get(), 129, 3, 2);
        run_l2_norm_case(backend.get(), 513, 2, 2);
        run_unary_case(backend.get(), GGML_UNARY_OP_SILU, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SIGMOID, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SOFTPLUS, 257);
        run_swiglu_case(backend.get(), 257);
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_F32, 17, 3, 2, 2.0e-4f, "mul_mat_vec_f32");
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_F16, 257, 3, 2, 2.0e-4f, "mul_mat_vec_f16");
        run_mul_mat_vec_case(backend.get(), dev, GGML_TYPE_BF16, 257, 3, 2, 2.0e-3f, "mul_mat_vec_bf16");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 17, 3, 2, 2, 1, 4, 3,
            3.0e-4f, "mul_mat_vec_f32_batched");
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 129, 2, 3, 1, 1, 3, 2,
            1.0e-3f, "mul_mat_vec_f16_batched");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q4_K, ggml_blck_size(GGML_TYPE_Q4_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q4_k_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q4_K, 2 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q4_k_two_blocks");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q5_k_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q5_K, 2 * ggml_blck_size(GGML_TYPE_Q5_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q5_k_two_blocks");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 3, 1,
            4.0e-4f, "mul_mat_vec_q6_k_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q6_K, 2 * ggml_blck_size(GGML_TYPE_Q6_K), 2, 3,
            5.0e-4f, "mul_mat_vec_q6_k_two_blocks");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, ggml_blck_size(GGML_TYPE_Q8_0), 3, 1,
            4.0e-4f, "mul_mat_vec_q8_0_one_block");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, 2 * ggml_blck_size(GGML_TYPE_Q8_0), 2, 3,
            5.0e-4f, "mul_mat_vec_q8_0_two_blocks");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_F16, GGML_TYPE_F16, true, true, "flash_attn_ext_f16");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_BF16, GGML_TYPE_BF16, false, false, "flash_attn_ext_bf16");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_F32, GGML_TYPE_F32, false, false, "flash_attn_ext_f32");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0, false, false, "flash_attn_ext_q4_0");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0, false, false, "flash_attn_ext_q8_0");
        run_flash_attn_ext_decode_case(
            backend.get(), dev, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0, false, false, "flash_attn_ext_q8_0_q4_0");
        run_soft_max_case(backend.get(), 1, false);
        run_soft_max_case(backend.get(), 257, false);
        run_soft_max_case(backend.get(), 257, true);
        run_rope_imrope_case(backend.get(), 12, 2, 3);
        run_rope_imrope_case(backend.get(), 128, 4, 2);
        run_gated_delta_net_case(backend.get(), false);
        run_gated_delta_net_case(backend.get(), true);
    }
    run_clamp_case(backend.get(), 1);
    run_clamp_case(backend.get(), 256);
    run_clamp_case(backend.get(), 257);

    ggml_backend_synchronize(backend.get());
    return 0;
}
