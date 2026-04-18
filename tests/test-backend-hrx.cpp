#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
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

static std::vector<float> reference_mul_mat_id_q4_k(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs) {
    std::vector<float> output(static_cast<size_t>(rows * n_ids * n_tokens), 0.0f);
    const int64_t rhs_cols = broadcast_rhs ? 1 : n_ids;
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id_pos = 0; id_pos < n_ids; ++id_pos) {
            const int32_t expert = ids[static_cast<size_t>(id_pos + n_ids * token)];
            GGML_ASSERT(expert >= 0 && expert < n_experts);
            const int64_t rhs_col = broadcast_rhs ? 0 : id_pos;
            for (int64_t row = 0; row < rows; ++row) {
                float sum = 0.0f;
                for (int64_t i = 0; i < k; ++i) {
                    const size_t lhs_idx = static_cast<size_t>(i + k * (row + rows * expert));
                    const size_t rhs_idx = static_cast<size_t>(i + k * (rhs_col + rhs_cols * token));
                    sum += lhs[lhs_idx] * rhs[rhs_idx];
                }
                output[static_cast<size_t>(row + rows * (id_pos + n_ids * token))] = sum;
            }
        }
    }
    return output;
}

static float reference_mul_mat_id_q4_k_value(
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        int64_t row,
        int64_t id_pos,
        int64_t token) {
    (void) rows;
    (void) n_tokens;
    const int32_t expert = ids[static_cast<size_t>(id_pos + n_ids * token)];
    GGML_ASSERT(expert >= 0 && expert < n_experts);
    const int64_t rhs_cols = broadcast_rhs ? 1 : n_ids;
    const int64_t rhs_col = broadcast_rhs ? 0 : id_pos;
    float sum = 0.0f;
    for (int64_t i = 0; i < k; ++i) {
        const size_t lhs_idx = static_cast<size_t>(i + k * (row + rows * expert));
        const size_t rhs_idx = static_cast<size_t>(i + k * (rhs_col + rhs_cols * token));
        sum += lhs[lhs_idx] * rhs[rhs_idx];
    }
    return sum;
}

static std::vector<int64_t> sample_positions(int64_t n) {
    std::vector<int64_t> positions = { 0 };
    if (n > 2) {
        positions.push_back(n / 2);
    }
    if (n > 1) {
        positions.push_back(n - 1);
    }
    return positions;
}

static void expect_mul_mat_id_q4_k_samples(
        const std::vector<float> & actual,
        const std::vector<float> & lhs,
        const std::vector<float> & rhs,
        const std::vector<int32_t> & ids,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        float tolerance,
        const char * label) {
    const std::vector<int64_t> row_samples = sample_positions(rows);
    const std::vector<int64_t> id_samples = sample_positions(n_ids);
    const std::vector<int64_t> token_samples = sample_positions(n_tokens);
    for (const int64_t token : token_samples) {
        for (const int64_t id_pos : id_samples) {
            for (const int64_t row : row_samples) {
                const size_t idx = static_cast<size_t>(row + rows * (id_pos + n_ids * token));
                const float expected = reference_mul_mat_id_q4_k_value(
                    lhs, rhs, ids, k, rows, n_ids, n_tokens, n_experts, broadcast_rhs, row, id_pos, token);
                const float got = actual[idx];
                if (std::fabs(got - expected) > tolerance) {
                    std::fprintf(stderr,
                        "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                        label,
                        static_cast<long long>(row),
                        static_cast<long long>(id_pos),
                        static_cast<long long>(token),
                        got,
                        expected,
                        tolerance);
                    std::abort();
                }
            }
        }
    }
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

static void run_mul_mat_id_q4_k_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens,
        int64_t n_experts,
        bool broadcast_rhs,
        float tolerance,
        const char * label,
        bool sampled_reference = false) {
    GGML_ASSERT(k % ggml_blck_size(GGML_TYPE_Q4_K) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, k, rows, n_experts);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, k, broadcast_rhs ? 1 : n_ids, n_tokens);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_ids, n_tokens);
    ggml_tensor * out = ggml_mul_mat_id(ctx.get(), lhs, rhs, ids);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows * n_experts));
    std::vector<float> rhs_f32(static_cast<size_t>(k * (broadcast_rhs ? 1 : n_ids) * n_tokens));
    std::vector<int32_t> ids_i32(static_cast<size_t>(n_ids * n_tokens));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 17) % 127) - 63) / 53.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 5) % 109) - 54) / 59.0f;
    }
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t id_pos = 0; id_pos < n_ids; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + n_ids * token)] =
                static_cast<int32_t>((id_pos * 3 + token * 5) % n_experts);
        }
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, k, rows * n_experts, lhs_f32, lhs_reference, lhs_storage);

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    if (sampled_reference) {
        expect_mul_mat_id_q4_k_samples(
            tensor_to_float(out),
            lhs_reference,
            rhs_f32,
            ids_i32,
            k,
            rows,
            n_ids,
            n_tokens,
            n_experts,
            broadcast_rhs,
            tolerance,
            label);
        return;
    }

    expect_near(
        tensor_to_float(out),
        reference_mul_mat_id_q4_k(
            lhs_reference, rhs_f32, ids_i32, k, rows, n_ids, n_tokens, n_experts, broadcast_rhs),
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

static void run_topk_moe_case(
        ggml_backend_t backend,
        int64_t n_experts,
        int64_t n_tokens,
        int64_t n_expert_used,
        bool with_norm,
        const char * label) {
    ggml_context_ptr ctx = make_context();
    int64_t ne[4] = { n_experts, n_tokens, 1, 1 };
    ggml_tensor * logits = ggml_new_tensor(ctx.get(), GGML_TYPE_F32, 4, ne);
    ggml_tensor * probs = ggml_soft_max(ctx.get(), logits);
    ggml_tensor * selected_experts = ggml_argsort_top_k(ctx.get(), probs, n_expert_used);
    ggml_tensor * weights = ggml_get_rows(
        ctx.get(), ggml_reshape_3d(ctx.get(), probs, 1, n_experts, n_tokens), selected_experts);
    if (with_norm) {
        weights = ggml_reshape_2d(ctx.get(), weights, n_expert_used, n_tokens);
        ggml_tensor * weights_sum = ggml_sum_rows(ctx.get(), weights);
        weights_sum = ggml_clamp(
            ctx.get(), weights_sum, 6.103515625e-5f, std::numeric_limits<float>::infinity());
        weights = ggml_div(ctx.get(), weights, weights_sum);
        weights = ggml_reshape_3d(ctx.get(), weights, 1, n_expert_used, n_tokens);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, weights);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> logits_data(static_cast<size_t>(n_experts * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            const int64_t raw = (expert * 37 + token * 19 + (expert / 5) * 3) % 29;
            float value = static_cast<float>(raw - 14) * 0.125f;
            if (token == 0 && expert < std::min<int64_t>(n_experts, 4)) {
                value = 1.25f;
            }
            logits_data[static_cast<size_t>(token * n_experts + expert)] = value;
        }
    }
    ggml_backend_tensor_set(logits, logits_data.data(), 0, logits_data.size() * sizeof(float));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<int32_t> expected_ids(static_cast<size_t>(n_expert_used * n_tokens));
    std::vector<float> expected_weights(static_cast<size_t>(n_expert_used * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        const float * row = logits_data.data() + token * n_experts;
        float max_value = row[0];
        for (int64_t expert = 1; expert < n_experts; ++expert) {
            max_value = std::max(max_value, row[expert]);
        }

        std::vector<float> probabilities(static_cast<size_t>(n_experts));
        float sum = 0.0f;
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            probabilities[static_cast<size_t>(expert)] = std::exp(row[expert] - max_value);
            sum += probabilities[static_cast<size_t>(expert)];
        }
        for (float & value : probabilities) {
            value /= sum;
        }

        std::vector<int32_t> order(static_cast<size_t>(n_experts));
        for (int64_t expert = 0; expert < n_experts; ++expert) {
            order[static_cast<size_t>(expert)] = static_cast<int32_t>(expert);
        }
        std::sort(order.begin(), order.end(), [&](int32_t lhs, int32_t rhs) {
            const float lhs_value = probabilities[static_cast<size_t>(lhs)];
            const float rhs_value = probabilities[static_cast<size_t>(rhs)];
            if (lhs_value != rhs_value) {
                return lhs_value > rhs_value;
            }
            return lhs < rhs;
        });

        float selected_sum = 0.0f;
        for (int64_t k = 0; k < n_expert_used; ++k) {
            selected_sum += probabilities[static_cast<size_t>(order[static_cast<size_t>(k)])];
        }
        const float denom = with_norm ? std::max(selected_sum, 6.103515625e-5f) : 1.0f;
        for (int64_t k = 0; k < n_expert_used; ++k) {
            const int32_t expert = order[static_cast<size_t>(k)];
            const size_t out_idx = static_cast<size_t>(token * n_expert_used + k);
            expected_ids[out_idx] = expert;
            expected_weights[out_idx] = probabilities[static_cast<size_t>(expert)] / denom;
        }
    }

    std::vector<int32_t> actual_ids(static_cast<size_t>(n_expert_used * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        ggml_backend_tensor_get(
            selected_experts,
            actual_ids.data() + token * n_expert_used,
            static_cast<size_t>(token) * selected_experts->nb[1],
            static_cast<size_t>(n_expert_used) * sizeof(int32_t));
    }
    GGML_ASSERT(actual_ids.size() == expected_ids.size());
    for (size_t i = 0; i < actual_ids.size(); ++i) {
        if (actual_ids[i] != expected_ids[i]) {
            std::fprintf(stderr, "%s ids[%zu]: got %" PRId32 " expected %" PRId32 "\n",
                label, i, actual_ids[i], expected_ids[i]);
            std::abort();
        }
    }
    expect_near(tensor_to_float(weights), expected_weights, 2.0e-5f, label);
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

class scoped_env_var {
public:
    scoped_env_var(const char * name, const char * value) : name(name) {
        if (const char * existing = std::getenv(name)) {
            old_value = existing;
            had_value = true;
        }
        ::setenv(name, value, 1);
    }

    ~scoped_env_var() {
        if (had_value) {
            ::setenv(name, old_value.c_str(), 1);
        } else {
            ::unsetenv(name);
        }
    }

private:
    const char * name;
    bool had_value = false;
    std::string old_value;
};

static void run_mul_mat_id_q4_k_mul_fusion_case(
        ggml_backend_t backend,
        const char * label,
        int64_t K = 256,
        int64_t ROWS = 8,
        int64_t N_IDS = 2,
        int64_t N_TOKENS = 3,
        int64_t N_EXPERTS = 4,
        float tolerance = 1.0e-3f,
        bool sampled_reference = false) {
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, N_IDS, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_IDS, N_TOKENS);
    ggml_tensor * scale = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, N_IDS, 1);
    ggml_tensor * mmid = ggml_mul_mat_id(ctx.get(), lhs, rhs, ids);
    ggml_tensor * out = ggml_mul(ctx.get(), mmid, scale);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS * N_EXPERTS));
    std::vector<float> rhs_f32(static_cast<size_t>(K * N_IDS * N_TOKENS));
    std::vector<float> scale_f32(static_cast<size_t>(N_IDS));
    std::vector<int32_t> ids_i32(static_cast<size_t>(N_IDS * N_TOKENS));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 13) % 97) - 48) / 53.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 7) % 89) - 44) / 47.0f;
    }
    for (int64_t i = 0; i < N_IDS; ++i) {
        scale_f32[static_cast<size_t>(i)] = 0.25f + 0.125f * static_cast<float>(i);
    }
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + N_IDS * token)] =
                static_cast<int32_t>((id_pos + 2 * token) % N_EXPERTS);
        }
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, lhs_f32, lhs_reference, lhs_storage);
    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));
    ggml_backend_tensor_set(scale, scale_f32.data(), 0, scale_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = tensor_to_float(out);
    if (sampled_reference) {
        const std::vector<int64_t> row_samples = sample_positions(ROWS);
        const std::vector<int64_t> id_samples = sample_positions(N_IDS);
        const std::vector<int64_t> token_samples = sample_positions(N_TOKENS);
        for (const int64_t token : token_samples) {
            for (const int64_t id_pos : id_samples) {
                for (const int64_t row : row_samples) {
                    const size_t idx = static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token));
                    const float expected = reference_mul_mat_id_q4_k_value(
                        lhs_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token) * scale_f32[static_cast<size_t>(id_pos)];
                    const float got = actual[idx];
                    if (std::fabs(got - expected) > tolerance) {
                        std::fprintf(stderr,
                            "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                            label,
                            static_cast<long long>(row),
                            static_cast<long long>(id_pos),
                            static_cast<long long>(token),
                            got,
                            expected,
                            tolerance);
                        std::abort();
                    }
                }
            }
        }
        return;
    }

    std::vector<float> expected =
        reference_mul_mat_id_q4_k(lhs_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            const float factor = scale_f32[static_cast<size_t>(id_pos)];
            for (int64_t row = 0; row < ROWS; ++row) {
                expected[static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token))] *= factor;
            }
        }
    }
    expect_near(actual, expected, tolerance, label);
}

static void run_mul_mat_id_q4_k_swiglu_fusion_case(
        ggml_backend_t backend,
        const char * label,
        int64_t K = 256,
        int64_t ROWS = 8,
        int64_t N_IDS = 2,
        int64_t N_TOKENS = 3,
        int64_t N_EXPERTS = 4,
        float tolerance = 2.0e-3f,
        bool sampled_reference = false) {
    scoped_env_var disable_swiglu("GGML_HRX_DISABLE_SWIGLU", "1");

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * up_lhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_Q4_K, K, ROWS, N_EXPERTS);
    ggml_tensor * rhs = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, K, N_IDS, N_TOKENS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, N_IDS, N_TOKENS);
    ggml_tensor * gate = ggml_mul_mat_id(ctx.get(), gate_lhs, rhs, ids);
    ggml_tensor * up = ggml_mul_mat_id(ctx.get(), up_lhs, rhs, ids);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(static_cast<size_t>(K * ROWS * N_EXPERTS));
    std::vector<float> up_f32(gate_f32.size());
    std::vector<float> rhs_f32(static_cast<size_t>(K * N_IDS * N_TOKENS));
    std::vector<int32_t> ids_i32(static_cast<size_t>(N_IDS * N_TOKENS));
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 5) % 101) - 50) / 67.0f;
        up_f32[i] = static_cast<float>(static_cast<int>((i * 29 + 11) % 103) - 51) / 71.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 31 + 3) % 107) - 53) / 73.0f;
    }
    for (int64_t token = 0; token < N_TOKENS; ++token) {
        for (int64_t id_pos = 0; id_pos < N_IDS; ++id_pos) {
            ids_i32[static_cast<size_t>(id_pos + N_IDS * token)] =
                static_cast<int32_t>((id_pos * 3 + token) % N_EXPERTS);
        }
    }

    std::vector<float> gate_reference;
    std::vector<float> up_reference;
    std::vector<uint8_t> gate_storage;
    std::vector<uint8_t> up_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, gate_f32, gate_reference, gate_storage);
    prepare_mul_mat_lhs(GGML_TYPE_Q4_K, K, ROWS * N_EXPERTS, up_f32, up_reference, up_storage);
    ggml_backend_tensor_set(gate_lhs, gate_storage.data(), 0, gate_storage.size());
    ggml_backend_tensor_set(up_lhs, up_storage.data(), 0, up_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_i32.data(), 0, ids_i32.size() * sizeof(int32_t));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    const std::vector<float> actual = tensor_to_float(out);
    if (sampled_reference) {
        const std::vector<int64_t> row_samples = sample_positions(ROWS);
        const std::vector<int64_t> id_samples = sample_positions(N_IDS);
        const std::vector<int64_t> token_samples = sample_positions(N_TOKENS);
        for (const int64_t token : token_samples) {
            for (const int64_t id_pos : id_samples) {
                for (const int64_t row : row_samples) {
                    const size_t idx = static_cast<size_t>(row + ROWS * (id_pos + N_IDS * token));
                    const float gate_expected = reference_mul_mat_id_q4_k_value(
                        gate_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token);
                    const float up_expected = reference_mul_mat_id_q4_k_value(
                        up_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false,
                        row, id_pos, token);
                    const float expected = gate_expected / (1.0f + std::exp(-gate_expected)) * up_expected;
                    const float got = actual[idx];
                    if (std::fabs(got - expected) > tolerance) {
                        std::fprintf(stderr,
                            "%s[%lld,%lld,%lld]: got %.9g expected %.9g tolerance %.9g\n",
                            label,
                            static_cast<long long>(row),
                            static_cast<long long>(id_pos),
                            static_cast<long long>(token),
                            got,
                            expected,
                            tolerance);
                        std::abort();
                    }
                }
            }
        }
        return;
    }

    std::vector<float> gate_expected =
        reference_mul_mat_id_q4_k(gate_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    std::vector<float> up_expected =
        reference_mul_mat_id_q4_k(up_reference, rhs_f32, ids_i32, K, ROWS, N_IDS, N_TOKENS, N_EXPERTS, false);
    for (size_t i = 0; i < gate_expected.size(); ++i) {
        gate_expected[i] = gate_expected[i] / (1.0f + std::exp(-gate_expected[i])) * up_expected[i];
    }
    expect_near(actual, gate_expected, tolerance, label);
}

static void run_bf16_mul_mat_swiglu_fusion_case(ggml_backend_t backend, const char * label) {
    scoped_env_var disable_swiglu("GGML_HRX_DISABLE_SWIGLU", "1");
    scoped_env_var disable_wmma("GGML_HRX_DISABLE_BF16_SWIGLU_WMMA16_PROMPT", "1");
    scoped_env_var disable_cols16("GGML_HRX_DISABLE_BF16_SWIGLU_COLS16_PROMPT", "1");

    static constexpr int64_t K = 256;
    static constexpr int64_t ROWS = 16;
    static constexpr int64_t COLS = 512;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * gate_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * up_lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, COLS);
    ggml_tensor * gate = ggml_mul_mat(ctx.get(), gate_lhs, rhs);
    ggml_tensor * up = ggml_mul_mat(ctx.get(), up_lhs, rhs);
    ggml_tensor * out = ggml_swiglu_split(ctx.get(), gate, up);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> gate_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> up_f32(gate_f32.size());
    std::vector<float> rhs_f32(static_cast<size_t>(K * COLS));
    for (size_t i = 0; i < gate_f32.size(); ++i) {
        gate_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 11) % 103) - 51) / 67.0f;
        up_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 5) % 107) - 53) / 71.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 23 + 3) % 109) - 54) / 73.0f;
    }

    std::vector<float> gate_reference;
    std::vector<float> up_reference;
    std::vector<uint8_t> gate_storage;
    std::vector<uint8_t> up_storage;
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, gate_f32, gate_reference, gate_storage);
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, up_f32, up_reference, up_storage);
    ggml_backend_tensor_set(gate_lhs, gate_storage.data(), 0, gate_storage.size());
    ggml_backend_tensor_set(up_lhs, up_storage.data(), 0, up_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> gate_expected = reference_mul_mat(gate_reference, rhs_f32, K, ROWS, COLS);
    std::vector<float> up_expected = reference_mul_mat(up_reference, rhs_f32, K, ROWS, COLS);
    for (size_t i = 0; i < gate_expected.size(); ++i) {
        gate_expected[i] = gate_expected[i] / (1.0f + std::exp(-gate_expected[i])) * up_expected[i];
    }
    expect_near(tensor_to_float(out), gate_expected, 3.0e-2f, label);
}

static void run_bf16_mul_mat_set_rows_fusion_case(ggml_backend_t backend, const char * label) {
    static constexpr int64_t K = 256;
    static constexpr int64_t ROWS = 2;
    static constexpr int64_t DST_ROWS = 4;

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_BF16, K, ROWS);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, K, 1);
    ggml_tensor * mul_mat = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * adapter = ggml_reshape_2d(ctx.get(), mul_mat, 1, ROWS);
    ggml_tensor * dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F16, 1, DST_ROWS);
    ggml_tensor * rows = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I64, ROWS);
    ggml_tensor * out = ggml_set_rows(ctx.get(), dst, adapter, rows);
    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(K * ROWS));
    std::vector<float> rhs_f32(static_cast<size_t>(K));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 97) - 48) / 61.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 19 + 3) % 89) - 44) / 59.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_BF16, K, ROWS, lhs_f32, lhs_reference, lhs_storage);
    const std::vector<uint8_t> dst_zero(ggml_nbytes(dst), 0);
    const int64_t row_ids[ROWS] = { 2, 0 };

    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(dst, dst_zero.data(), 0, dst_zero.size());
    ggml_backend_tensor_set(rows, row_ids, 0, sizeof(row_ids));
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected(DST_ROWS, 0.0f);
    const std::vector<float> matmul = reference_mul_mat(lhs_reference, rhs_f32, K, ROWS, 1);
    for (int64_t row = 0; row < ROWS; ++row) {
        expected[static_cast<size_t>(row_ids[row])] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(matmul[row]));
    }
    expect_near(tensor_to_float(out), expected, 0.0f, label);
}

static void run_q8_0_mul_mat_add_fusion_case(
        ggml_backend_t backend,
        int64_t k,
        int64_t rows,
        int64_t cols,
        float tolerance,
        const char * label) {
    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(k % ggml_blck_size(GGML_TYPE_Q8_0) == 0);

    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_Q8_0, k, rows);
    ggml_tensor * rhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, k, cols);
    ggml_tensor * bias = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, rows, cols);
    ggml_tensor * mm = ggml_mul_mat(ctx.get(), lhs, rhs);
    ggml_tensor * out = ggml_add(ctx.get(), mm, bias);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);
    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_f32(static_cast<size_t>(k * rows));
    std::vector<float> rhs_f32(static_cast<size_t>(k * cols));
    std::vector<float> bias_f32(static_cast<size_t>(rows * cols));
    for (size_t i = 0; i < lhs_f32.size(); ++i) {
        lhs_f32[i] = static_cast<float>(static_cast<int>((i * 17 + 7) % 101) - 50) / 39.0f;
    }
    for (size_t i = 0; i < rhs_f32.size(); ++i) {
        rhs_f32[i] = static_cast<float>(static_cast<int>((i * 13 + 5) % 89) - 44) / 43.0f;
    }
    for (size_t i = 0; i < bias_f32.size(); ++i) {
        bias_f32[i] = static_cast<float>(static_cast<int>((i * 11 + 3) % 31) - 15) / 29.0f;
    }

    std::vector<float> lhs_reference;
    std::vector<uint8_t> lhs_storage;
    prepare_mul_mat_lhs(GGML_TYPE_Q8_0, k, rows, lhs_f32, lhs_reference, lhs_storage);
    ggml_backend_tensor_set(lhs, lhs_storage.data(), 0, lhs_storage.size());
    ggml_backend_tensor_set(rhs, rhs_f32.data(), 0, rhs_f32.size() * sizeof(float));
    ggml_backend_tensor_set(bias, bias_f32.data(), 0, bias_f32.size() * sizeof(float));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> expected = reference_mul_mat(lhs_reference, rhs_f32, k, rows, cols);
    for (size_t i = 0; i < expected.size(); ++i) {
        expected[i] += bias_f32[i];
    }
    expect_near(tensor_to_float(out), expected, tolerance, label);
}

static void expect_flash_attn_ext_prefill_samples(
        const std::vector<float> & actual,
        const std::vector<float> & q,
        const std::vector<float> & k,
        const std::vector<float> & v,
        const std::vector<float> & mask,
        float scale,
        float logit_softcap,
        const char * label) {
    static constexpr int64_t D = 256;
    static constexpr int64_t N = 512;
    static constexpr int64_t H = 16;
    static constexpr int64_t H_KV = 2;
    static constexpr int64_t KV = 512;
    const std::array<int64_t, 5> sample_tokens = { 0, 1, 17, 255, 511 };
    const std::array<int64_t, 3> sample_heads = { 0, 7, 15 };
    const std::array<int64_t, 4> sample_cols = { 0, 31, 128, 255 };
    std::vector<float> logits(static_cast<size_t>(KV));
    std::vector<float> expected(static_cast<size_t>(D));

    for (const int64_t token : sample_tokens) {
        for (const int64_t head : sample_heads) {
            const int64_t kv_head = head / (H / H_KV);
            float max_score = -INFINITY;
            for (int64_t t = 0; t < KV; ++t) {
                float score = 0.0f;
                for (int64_t col = 0; col < D; ++col) {
                    score +=
                        q[index_4d(col, token, head, 0, D, N, H)] *
                        k[index_4d(col, t, kv_head, 0, D, KV, H_KV)];
                }
                score *= scale;
                if (logit_softcap != 0.0f) {
                    score = logit_softcap * std::tanh(score / logit_softcap);
                }
                score += mask[index_4d(t, token, 0, 0, KV, N, 1)];
                logits[static_cast<size_t>(t)] = score;
                max_score = std::max(max_score, score);
            }

            float denom = 0.0f;
            for (float & logit : logits) {
                logit = std::exp(logit - max_score);
                denom += logit;
            }

            for (int64_t col = 0; col < D; ++col) {
                float value = 0.0f;
                for (int64_t t = 0; t < KV; ++t) {
                    value += logits[static_cast<size_t>(t)] *
                        v[index_4d(col, t, kv_head, 0, D, KV, H_KV)];
                }
                expected[static_cast<size_t>(col)] = value / denom;
            }

            for (const int64_t col : sample_cols) {
                const size_t out_idx = index_4d(col, head, token, 0, D, H, N);
                const float got = actual[out_idx];
                const float want = expected[static_cast<size_t>(col)];
                if (std::fabs(got - want) > 2.0e-2f) {
                    std::fprintf(stderr,
                        "%s[col=%" PRId64 ",head=%" PRId64 ",token=%" PRId64 "]: got %.9g expected %.9g\n",
                        label, col, head, token, got, want);
                    std::abort();
                }
            }
        }
    }
}

static void run_flash_attn_ext_prefill_f16_case(
        ggml_backend_t backend,
        ggml_backend_dev_t dev,
        bool disable_direct,
        bool disable_wmma,
        const char * label,
        float logit_softcap = 0.0f) {
    scoped_env_var disable_decode("GGML_HRX_DISABLE_FLASH_ATTN_EXT_DECODE", "1");
    scoped_env_var disable_direct_var(
        "GGML_HRX_DISABLE_F16_PREFILL_FA_DIRECT", disable_direct ? "1" : "0");
    scoped_env_var disable_wmma_var("GGML_HRX_DISABLE_F16_PREFILL_FA_WMMA", disable_wmma ? "1" : "0");
    scoped_env_var disable_tile("GGML_HRX_DISABLE_F16_PREFILL_FA_TILE", "0");

    static constexpr int64_t D = 256;
    static constexpr int64_t N = 512;
    static constexpr int64_t H = 16;
    static constexpr int64_t H_KV = 2;
    static constexpr int64_t KV = 512;
    static constexpr int64_t S = 1;
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    ggml_context_ptr ctx = make_context();
    ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, D, N, H, S);
    ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, D, KV, H_KV, S);
    ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, D, KV, H_KV, S);
    ggml_tensor * mask = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F16, KV, N, 1, S);
    ggml_tensor * out = ggml_flash_attn_ext(ctx.get(), q, k, v, mask, scale, 0.0f, logit_softcap);
    GGML_ASSERT(ggml_backend_dev_supports_op(dev, out));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> q_data(static_cast<size_t>(D * N * H * S));
    std::vector<float> k_data(static_cast<size_t>(D * KV * H_KV * S));
    std::vector<float> v_data(static_cast<size_t>(D * KV * H_KV * S));
    for (size_t i = 0; i < q_data.size(); ++i) {
        q_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 41) - 20) / 61.0f;
    }
    for (size_t i = 0; i < k_data.size(); ++i) {
        k_data[i] = static_cast<float>(static_cast<int>((i * 11 + 5) % 37) - 18) / 67.0f;
    }
    for (size_t i = 0; i < v_data.size(); ++i) {
        v_data[i] = static_cast<float>(static_cast<int>((i * 13 + 9) % 43) - 21) / 59.0f;
    }

    std::vector<float> k_reference;
    std::vector<float> v_reference;
    std::vector<uint8_t> k_storage;
    std::vector<uint8_t> v_storage;
    prepare_rows(GGML_TYPE_F16, D, KV * H_KV * S, k_data, k_reference, k_storage);
    prepare_rows(GGML_TYPE_F16, D, KV * H_KV * S, v_data, v_reference, v_storage);

    std::vector<float> mask_reference(static_cast<size_t>(KV * N * S));
    std::vector<ggml_fp16_t> mask_storage(mask_reference.size());
    for (int64_t token = 0; token < N; ++token) {
        for (int64_t t = 0; t < KV; ++t) {
            const float value = t > token ? -1000.0f : 0.03125f * static_cast<float>(token - t);
            const size_t idx = index_4d(t, token, 0, 0, KV, N, 1);
            mask_reference[idx] = ggml_fp16_to_fp32(ggml_fp32_to_fp16(value));
            mask_storage[idx] = ggml_fp32_to_fp16(value);
        }
    }

    ggml_backend_tensor_set(q, q_data.data(), 0, q_data.size() * sizeof(float));
    ggml_backend_tensor_set(k, k_storage.data(), 0, k_storage.size());
    ggml_backend_tensor_set(v, v_storage.data(), 0, v_storage.size());
    ggml_backend_tensor_set(mask, mask_storage.data(), 0, mask_storage.size() * sizeof(ggml_fp16_t));

    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_flash_attn_ext_prefill_samples(
        tensor_to_float(out), q_data, k_reference, v_reference, mask_reference, scale, logit_softcap, label);
}

static float ssm_conv_update_value(
        const std::vector<float> & conv_state,
        const std::vector<float> & input,
        int64_t state_width,
        int64_t n_tokens,
        int64_t channel,
        int64_t logical_pos) {
    if (logical_pos < state_width) {
        return conv_state[static_cast<size_t>(channel * state_width + logical_pos)];
    }
    return input[static_cast<size_t>(channel * n_tokens + logical_pos - state_width)];
}

static void run_ssm_conv_update_fusion_case(
        ggml_backend_t backend,
        int64_t d_conv,
        int64_t d_inner,
        int64_t n_tokens,
        bool apply_silu,
        bool strided_conv_state = false) {
    ggml_context_ptr ctx = make_context();
    const int64_t state_width = d_conv - 1;
    ggml_tensor * conv_state_base = strided_conv_state ?
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_inner, state_width) : nullptr;
    ggml_tensor * conv_state = strided_conv_state ?
        ggml_transpose(ctx.get(), conv_state_base) :
        ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, state_width, d_inner);
    if (strided_conv_state) {
        GGML_ASSERT(conv_state->nb[0] != sizeof(float));
    }
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n_tokens, d_inner);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, d_conv, d_inner);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(
        ctx.get(), concat, state_width, d_inner, concat->nb[1], static_cast<size_t>(n_tokens) * sizeof(float));
    ggml_tensor * state_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, state_width, d_inner);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * ssm = ggml_ssm_conv(ctx.get(), concat, weight);
    ggml_tensor * out = apply_silu ? ggml_silu(ctx.get(), ssm) : ssm;

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> conv_state_data(static_cast<size_t>(state_width * d_inner));
    std::vector<float> input_data(static_cast<size_t>(n_tokens * d_inner));
    std::vector<float> weight_data(static_cast<size_t>(d_conv * d_inner));
    for (size_t i = 0; i < conv_state_data.size(); ++i) {
        conv_state_data[i] = static_cast<float>((i * 5) % 17) * 0.07f - 0.45f;
    }
    for (size_t i = 0; i < input_data.size(); ++i) {
        input_data[i] = static_cast<float>((i * 7) % 23) * 0.05f - 0.5f;
    }
    for (size_t i = 0; i < weight_data.size(); ++i) {
        weight_data[i] = static_cast<float>((i * 11) % 19) * 0.04f - 0.35f;
    }
    if (strided_conv_state) {
        std::vector<float> conv_state_base_data(static_cast<size_t>(state_width * d_inner));
        for (int64_t channel = 0; channel < d_inner; ++channel) {
            for (int64_t i = 0; i < state_width; ++i) {
                conv_state_base_data[static_cast<size_t>(i * d_inner + channel)] =
                    conv_state_data[static_cast<size_t>(channel * state_width + i)];
            }
        }
        ggml_backend_tensor_set(
            conv_state_base, conv_state_base_data.data(), 0, conv_state_base_data.size() * sizeof(float));
    } else {
        ggml_backend_tensor_set(conv_state, conv_state_data.data(), 0, conv_state_data.size() * sizeof(float));
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);

    std::vector<float> expected_out(static_cast<size_t>(d_inner * n_tokens));
    for (int64_t token = 0; token < n_tokens; ++token) {
        for (int64_t channel = 0; channel < d_inner; ++channel) {
            float sum = 0.0f;
            for (int64_t i = 0; i < d_conv; ++i) {
                const float x = ssm_conv_update_value(
                    conv_state_data, input_data, state_width, n_tokens, channel, token + i);
                const float w = weight_data[static_cast<size_t>(channel * d_conv + i)];
                sum += x * w;
            }
            if (apply_silu) {
                sum = sum / (1.0f + std::exp(-sum));
            }
            expected_out[static_cast<size_t>(token * d_inner + channel)] = sum;
        }
    }

    std::vector<float> expected_state(static_cast<size_t>(state_width * d_inner));
    for (int64_t channel = 0; channel < d_inner; ++channel) {
        for (int64_t i = 0; i < state_width; ++i) {
            expected_state[static_cast<size_t>(channel * state_width + i)] = ssm_conv_update_value(
                conv_state_data, input_data, state_width, n_tokens, channel, n_tokens + i);
        }
    }

    const char * out_label = strided_conv_state ?
        (apply_silu ? "ssm_conv_update_strided_silu_fusion_out" : "ssm_conv_update_strided_fusion_out") :
        (apply_silu ? "ssm_conv_update_silu_fusion_out" : "ssm_conv_update_fusion_out");
    const char * state_label = strided_conv_state ?
        (apply_silu ? "ssm_conv_update_strided_silu_fusion_state" : "ssm_conv_update_strided_fusion_state") :
        (apply_silu ? "ssm_conv_update_silu_fusion_state" : "ssm_conv_update_fusion_state");
    expect_near(tensor_to_float(out), expected_out, apply_silu ? 2.0e-5f : 2.0e-6f, out_label);
    expect_near(tensor_to_float(state_update), expected_state, 0.0f, state_label);
}

static void run_ssm_conv_update_negative_state_offset_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t D_CONV = 4;
    constexpr int64_t STATE_WIDTH = D_CONV - 1;
    constexpr int64_t D_INNER = 5;
    constexpr int64_t N_TOKENS = 4;
    ggml_tensor * conv_state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_TOKENS, D_INNER);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D_CONV, D_INNER);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(ctx.get(), concat, STATE_WIDTH, D_INNER, concat->nb[1], 0);
    ggml_tensor * state_dst = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, state_dst);
    ggml_tensor * out = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), concat, weight));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> data(static_cast<size_t>(N_TOKENS * D_INNER), 0.25f);
    std::vector<float> state_data(static_cast<size_t>(STATE_WIDTH * D_INNER), -0.125f);
    std::vector<float> weight_data(static_cast<size_t>(D_CONV * D_INNER), 0.0625f);
    ggml_backend_tensor_set(conv_state, state_data.data(), 0, state_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_ssm_conv_update_negative_state_overlap_case(ggml_backend_t backend) {
    ggml_context_ptr ctx = make_context();
    constexpr int64_t D_CONV = 4;
    constexpr int64_t STATE_WIDTH = D_CONV - 1;
    constexpr int64_t D_INNER = 5;
    constexpr int64_t N_TOKENS = 4;
    ggml_tensor * conv_state = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, STATE_WIDTH, D_INNER);
    ggml_tensor * input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, N_TOKENS, D_INNER);
    ggml_tensor * weight = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, D_CONV, D_INNER);
    ggml_tensor * concat = ggml_concat(ctx.get(), conv_state, input, 0);
    ggml_tensor * state_view = ggml_view_2d(
        ctx.get(), concat, STATE_WIDTH, D_INNER, concat->nb[1], static_cast<size_t>(N_TOKENS) * sizeof(float));
    ggml_tensor * state_update = ggml_cpy(ctx.get(), state_view, conv_state);
    ggml_tensor * out = ggml_silu(ctx.get(), ggml_ssm_conv(ctx.get(), concat, weight));

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, state_update);
    ggml_build_forward_expand(graph, out);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> data(static_cast<size_t>(N_TOKENS * D_INNER), 0.25f);
    std::vector<float> state_data(static_cast<size_t>(STATE_WIDTH * D_INNER), -0.125f);
    std::vector<float> weight_data(static_cast<size_t>(D_CONV * D_INNER), 0.0625f);
    ggml_backend_tensor_set(conv_state, state_data.data(), 0, state_data.size() * sizeof(float));
    ggml_backend_tensor_set(input, data.data(), 0, data.size() * sizeof(float));
    ggml_backend_tensor_set(weight, weight_data.data(), 0, weight_data.size() * sizeof(float));

    scoped_env_var disable_concat("GGML_HRX_DISABLE_CONCAT", "1");
    scoped_env_var disable_ssm_conv("GGML_HRX_DISABLE_SSM_CONV", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS);
}

static void run_add8_fusion_case(ggml_backend_t backend, int64_t ncols, int64_t nrows) {
    ggml_context_ptr ctx = make_context();
    static constexpr int ARITY = 8;
    const int64_t base_cols = ncols + 3;
    std::array<ggml_tensor *, ARITY> bases = {};
    std::array<ggml_tensor *, ARITY> views = {};
    for (int i = 0; i < ARITY; ++i) {
        bases[i] = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, base_cols, nrows);
        views[i] = ggml_view_2d(
            ctx.get(), bases[i], ncols, nrows, bases[i]->nb[1], static_cast<size_t>(i % 3) * sizeof(float));
    }

    ggml_tensor * sum = ggml_add(ctx.get(), views[0], views[1]);
    for (int i = 2; i < ARITY; ++i) {
        sum = ggml_add(ctx.get(), sum, views[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    for (int i = 0; i < ARITY; ++i) {
        ggml_build_forward_expand(graph, views[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<std::vector<float>> base_data;
    base_data.reserve(ARITY);
    std::vector<float> expected(static_cast<size_t>(ncols * nrows), 0.0f);
    for (int i = 0; i < ARITY; ++i) {
        base_data.emplace_back(static_cast<size_t>(base_cols * nrows));
        for (int64_t row = 0; row < nrows; ++row) {
            for (int64_t col = 0; col < base_cols; ++col) {
                base_data.back()[static_cast<size_t>(col + base_cols * row)] =
                    static_cast<float>(((col + 7 * row + 11 * i) % 29) - 14) / 17.0f;
            }
            for (int64_t col = 0; col < ncols; ++col) {
                expected[static_cast<size_t>(col + ncols * row)] +=
                    base_data.back()[static_cast<size_t>(col + (i % 3) + base_cols * row)];
            }
        }
        ggml_backend_tensor_set(bases[i], base_data.back().data(), 0, base_data.back().size() * sizeof(float));
    }

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(sum), expected, 1.0e-6f, "add8_fusion");
}

static void run_mul_sum8_fusion_case(ggml_backend_t backend, int64_t rows, int64_t tokens) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }

    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 32, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> values_data(static_cast<size_t>(rows * 8 * tokens));
    std::vector<float> scales_data(static_cast<size_t>(8 * tokens));
    for (size_t i = 0; i < values_data.size(); ++i) {
        values_data[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) / 19.0f;
    }
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales_data[i] = static_cast<float>(static_cast<int>((i * 7 + 2) % 23) - 11) / 13.0f;
    }

    std::vector<float> expected(static_cast<size_t>(rows * tokens), 0.0f);
    for (int64_t token = 0; token < tokens; ++token) {
        for (int64_t row = 0; row < rows; ++row) {
            float acc = 0.0f;
            for (int64_t expert = 0; expert < 8; ++expert) {
                const size_t value_idx = static_cast<size_t>(row + rows * (expert + 8 * token));
                const size_t scale_idx = static_cast<size_t>(expert + 8 * token);
                acc += values_data[value_idx] * scales_data[scale_idx];
            }
            expected[static_cast<size_t>(row + rows * token)] = acc;
        }
    }

    ggml_backend_tensor_set(values, values_data.data(), 0, values_data.size() * sizeof(float));
    ggml_backend_tensor_set(scales, scales_data.data(), 0, scales_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(sum), expected, 2.0e-6f, "mul_sum8_fusion");
}

static void set_mul_sum8_inputs(ggml_tensor * values, ggml_tensor * scales, int64_t rows, int64_t tokens) {
    std::vector<float> values_data(static_cast<size_t>(rows * 8 * tokens));
    std::vector<float> scales_data(static_cast<size_t>(8 * tokens));
    for (size_t i = 0; i < values_data.size(); ++i) {
        values_data[i] = static_cast<float>(static_cast<int>((i * 5 + 3) % 37) - 18) / 19.0f;
    }
    for (size_t i = 0; i < scales_data.size(); ++i) {
        scales_data[i] = static_cast<float>(static_cast<int>((i * 7 + 2) % 23) - 11) / 13.0f;
    }
    ggml_backend_tensor_set(values, values_data.data(), 0, values_data.size() * sizeof(float));
    ggml_backend_tensor_set(scales, scales_data.data(), 0, scales_data.size() * sizeof(float));
}

static void expect_graph_failure(ggml_backend_t backend, ggml_cgraph * graph, const char * label) {
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    if (status == GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s unexpectedly succeeded\n", label);
        std::abort();
    }
}

static void run_mul_sum8_negative_intervening_op_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);
    ggml_tensor * intervening_add = ggml_add(ctx.get(), scales, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }
    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    ggml_build_forward_expand(graph, intervening_add);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    expect_graph_failure(backend, graph, "mul_sum8_intervening_op");
}

static void run_mul_sum8_negative_extra_product_consumer_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    for (int i = 0; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }
    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }
    ggml_tensor * extra_consumer = ggml_scale_bias(ctx.get(), product, 1.0f, 0.0f);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);
    ggml_build_forward_expand(graph, extra_consumer);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    expect_graph_failure(backend, graph, "mul_sum8_extra_product_consumer");
}

static void run_mul_sum8_negative_duplicate_slice_case(ggml_backend_t backend) {
    static constexpr int64_t rows = 17;
    static constexpr int64_t tokens = 2;
    ggml_context_ptr ctx = make_context();
    ggml_tensor * values = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, rows, 8, tokens);
    ggml_tensor * scales = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 1, 8, tokens);
    ggml_tensor * product = ggml_mul(ctx.get(), values, scales);

    std::array<ggml_tensor *, 8> slices = {};
    slices[0] = ggml_view_2d(ctx.get(), product, rows, tokens, product->nb[2], 0);
    slices[1] = ggml_view_2d(ctx.get(), product, rows, tokens, product->nb[2], 0);
    for (int i = 2; i < 8; ++i) {
        slices[i] = ggml_view_2d(
            ctx.get(), product, rows, tokens, product->nb[2], static_cast<size_t>(i) * product->nb[1]);
    }

    ggml_tensor * sum = ggml_add(ctx.get(), slices[0], slices[1]);
    for (int i = 2; i < 8; ++i) {
        sum = ggml_add(ctx.get(), sum, slices[i]);
    }

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 40, false);
    ggml_build_forward_expand(graph, product);
    for (int i = 0; i < 8; ++i) {
        ggml_build_forward_expand(graph, slices[i]);
    }
    ggml_build_forward_expand(graph, sum);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);
    set_mul_sum8_inputs(values, scales, rows, tokens);

    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    expect_graph_failure(backend, graph, "mul_sum8_duplicate_slice");
}

static void run_mul_add_add_fusion_case(ggml_backend_t backend, int64_t ncols, int64_t nrows) {
    ggml_context_ptr ctx = make_context();
    ggml_tensor * lhs = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * scale = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 1, nrows);
    ggml_tensor * bias0 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, 1);
    ggml_tensor * bias1 = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * product = ggml_mul(ctx.get(), lhs, scale);
    ggml_tensor * first_add = ggml_add(ctx.get(), product, bias0);
    ggml_tensor * second_add = ggml_add(ctx.get(), first_add, bias1);

    ggml_cgraph * graph = ggml_new_graph_custom(ctx.get(), 16, false);
    ggml_build_forward_expand(graph, second_add);

    ggml_backend_buffer_ptr buffer(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
    GGML_ASSERT(buffer != nullptr);

    std::vector<float> lhs_data(static_cast<size_t>(ncols * nrows));
    std::vector<float> scale_data(static_cast<size_t>(nrows));
    std::vector<float> bias0_data(static_cast<size_t>(ncols));
    std::vector<float> bias1_data(static_cast<size_t>(ncols * nrows));
    for (size_t i = 0; i < lhs_data.size(); ++i) {
        lhs_data[i] = static_cast<float>(static_cast<int>((i * 13 + 1) % 41) - 20) / 23.0f;
    }
    for (size_t i = 0; i < scale_data.size(); ++i) {
        scale_data[i] = static_cast<float>(static_cast<int>((i * 3 + 5) % 17) - 8) / 11.0f;
    }
    for (size_t i = 0; i < bias0_data.size(); ++i) {
        bias0_data[i] = static_cast<float>(static_cast<int>((i * 7 + 3) % 19) - 9) / 29.0f;
    }
    for (size_t i = 0; i < bias1_data.size(); ++i) {
        bias1_data[i] = static_cast<float>(static_cast<int>((i * 11 + 4) % 31) - 15) / 31.0f;
    }

    std::vector<float> expected(lhs_data.size());
    for (int64_t row = 0; row < nrows; ++row) {
        for (int64_t col = 0; col < ncols; ++col) {
            const size_t idx = static_cast<size_t>(col + ncols * row);
            expected[idx] = lhs_data[idx] * scale_data[static_cast<size_t>(row)] +
                bias0_data[static_cast<size_t>(col)] + bias1_data[idx];
        }
    }

    ggml_backend_tensor_set(lhs, lhs_data.data(), 0, lhs_data.size() * sizeof(float));
    ggml_backend_tensor_set(scale, scale_data.data(), 0, scale_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias0, bias0_data.data(), 0, bias0_data.size() * sizeof(float));
    ggml_backend_tensor_set(bias1, bias1_data.data(), 0, bias1_data.size() * sizeof(float));

    scoped_env_var disable_add("GGML_HRX_DISABLE_ADD", "1");
    scoped_env_var disable_mul("GGML_HRX_DISABLE_MUL", "1");
    GGML_ASSERT(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    expect_near(tensor_to_float(second_add), expected, 1.0e-6f, "mul_add_add_fusion");
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
    run_ssm_conv_update_fusion_case(backend.get(), 4, 33, 17, false);
    run_ssm_conv_update_fusion_case(backend.get(), 4, 33, 17, false, true);
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
        run_add8_fusion_case(backend.get(), 1, 1);
        run_add8_fusion_case(backend.get(), 257, 3);
        run_mul_sum8_fusion_case(backend.get(), 257, 3);
        run_mul_sum8_negative_intervening_op_case(backend.get());
        run_mul_sum8_negative_extra_product_consumer_case(backend.get());
        run_mul_sum8_negative_duplicate_slice_case(backend.get());
        run_mul_add_add_fusion_case(backend.get(), 1, 1);
        run_mul_add_add_fusion_case(backend.get(), 257, 3);
        run_ssm_conv_update_fusion_case(backend.get(), 4, 33, 17, true);
        run_ssm_conv_update_fusion_case(backend.get(), 4, 33, 17, true, true);
        run_ssm_conv_update_negative_state_offset_case(backend.get());
        run_ssm_conv_update_negative_state_overlap_case(backend.get());
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
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q4_K, ggml_blck_size(GGML_TYPE_Q4_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q4_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q4_K, 2 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q4_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q5_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, 2 * ggml_blck_size(GGML_TYPE_Q5_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 32, 32,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmq32");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 64, 64,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmq64");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 128, 128,
                1.0e-1f, "mul_mat_vec_q5_k_q8_1_x4_mmql128");
            {
                scoped_env_var disable_q5_x4_mmql128("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128", "1");
                scoped_env_var disable_q5_x4_mmq64("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ64", "1");
                scoped_env_var disable_q5_x4_mmq32("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ32", "1");
                run_mul_mat_vec_case(
                    backend.get(), dev, GGML_TYPE_Q5_K, ggml_blck_size(GGML_TYPE_Q5_K), 32, 32,
                    1.0e-1f, "mul_mat_vec_q5_k_q8_1_mmq32");
            }
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 3, 1,
                8.0e-2f, "mul_mat_vec_q6_k_q8_1_one_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, 2 * ggml_blck_size(GGML_TYPE_Q6_K), 2, 3,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_multi_col");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 32, 32,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmq32");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q6_K, ggml_blck_size(GGML_TYPE_Q6_K), 128, 64,
                1.0e-1f, "mul_mat_vec_q6_k_q8_1_x4_mmql128");
        }
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
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_Q8_0, 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 512,
            5.0e-4f, "mul_mat_vec_q8_0_cols8_prompt");
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_Q8_0, 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 512,
                1.0e-1f, "mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt");
        }
        run_q8_0_mul_mat_add_fusion_case(
            backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 1,
            5.0e-4f, "mul_mat_vec_q8_0_add_scalar");
        {
            scoped_env_var disable_x4("GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT", "1");
            scoped_env_var disable_rows4("GGML_HRX_DISABLE_Q8_0_ADD_ROWS4_COLS4_PROMPT", "1");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 3, 512,
                5.0e-4f, "mul_mat_vec_q8_0_add_cols8_prompt");
        }
        {
            scoped_env_var disable_x4("GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT", "1");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 2 * ggml_blck_size(GGML_TYPE_Q8_0), 4, 512,
                5.0e-4f, "mul_mat_vec_q8_0_add_rows4_cols4_prompt");
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_q8_0_mul_mat_add_fusion_case(
                backend.get(), 4 * ggml_blck_size(GGML_TYPE_Q8_0), 128, 512,
                1.0e-1f, "mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_prompt");
        }
        {
            scoped_env_var disable_wmma("GGML_HRX_DISABLE_BF16_WMMA16_PROMPT", "1");
            scoped_env_var disable_cols32("GGML_HRX_DISABLE_BF16_COLS32_PROMPT", "1");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_BF16, 256, 16, 512,
                3.0e-2f, "mul_mat_vec_bf16_cols16_prompt");
        }
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_BF16, 256, 16, 512,
            6.0e-2f, "mul_mat_vec_bf16_wmma16_prompt");
        run_mul_mat_vec_case(
            backend.get(), dev, GGML_TYPE_BF16, 512, 2048, 1,
            3.0e-2f, "mul_mat_vec_bf16_rows4_k512_decode");
        {
            scoped_env_var force_bf16_wg128("GGML_HRX_MUL_MAT_VEC_BF16_WORKGROUP_SIZE", "128");
            run_mul_mat_vec_case(
                backend.get(), dev, GGML_TYPE_BF16, 257, 3, 2,
                2.0e-3f, "mul_mat_vec_bf16_wg128");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F16, 256, 16, 512, 2, 1, 2, 1,
            2.0e-2f, "mul_mat_vec_f16_batched_cols16_prompt");
        {
            scoped_env_var disable_f16_cols16("GGML_HRX_DISABLE_F16_BATCHED_COLS16_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F16, 256, 16, 512, 2, 1, 2, 1,
                2.0e-2f, "mul_mat_vec_f16_batched_cols8_prompt");
        }
        {
            scoped_env_var disable_f16_cols16("GGML_HRX_DISABLE_F16_BATCHED_COLS16_PROMPT", "1");
            scoped_env_var disable_f16_cols8("GGML_HRX_DISABLE_F16_BATCHED_COLS8_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F16, 256, 16, 512, 2, 1, 2, 1,
                2.0e-2f, "mul_mat_vec_f16_batched_cols4_prompt");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 256, 16, 512, 2, 1, 2, 1,
            1.0e-3f, "mul_mat_vec_f32_batched_rows2_cols8_prompt");
        {
            scoped_env_var disable_f32_rows2("GGML_HRX_DISABLE_F32_BATCHED_ROWS2_COLS8_PROMPT", "1");
            run_mul_mat_vec_batched_case(
                backend.get(), dev, GGML_TYPE_F32, 256, 16, 512, 2, 1, 2, 1,
                1.0e-3f, "mul_mat_vec_f32_batched_cols16_prompt");
        }
        run_mul_mat_vec_batched_case(
            backend.get(), dev, GGML_TYPE_F32, 2048, 4, 1, 1, 1, 1, 1,
            2.0e-3f, "mul_mat_vec_f32_batched_cols1_k2048");
        run_bf16_mul_mat_swiglu_fusion_case(backend.get(), "mul_mat_vec_bf16_swiglu_fusion");
        run_bf16_mul_mat_set_rows_fusion_case(backend.get(), "mul_mat_vec_bf16_set_rows_fusion");
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, ggml_blck_size(GGML_TYPE_Q4_K), 3, 2, 3, 4, false,
            6.0e-4f, "mul_mat_id_q4_k_wg64");
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, 9 * ggml_blck_size(GGML_TYPE_Q4_K), 2, 3, 2, 5, true,
            3.0e-3f, "mul_mat_id_q4_k_wg256_broadcast");
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            scoped_env_var disable_grouped("GGML_HRX_DISABLE_Q4_K_ID_GROUPED_PROMPT", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 4, 1, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_row4_prompt", true);
        }
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 2, 8, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_grouped_row2_route8_prompt", true);
        }
        run_mul_mat_id_q4_k_case(
            backend.get(), dev, 512, 64, 8, 512, 4, false,
            2.0e-1f, "mul_mat_id_q4_k_grouped_q8_1_x4_prompt", true);
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 512, 64, 8, 512, 4, false,
                2.0e-3f, "mul_mat_id_q4_k_grouped_q8_1_disabled_prompt", true);
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_id_q4_k_case(
                backend.get(), dev, 256, 8, 2, 3, 4, false,
                1.0e-1f, "mul_mat_id_q4_k_q8_1_forced");
        }
        run_mul_mat_id_q4_k_mul_fusion_case(backend.get(), "mul_mat_id_q4_k_mul_fusion");
        run_mul_mat_id_q4_k_mul_fusion_case(
            backend.get(), "mul_mat_id_q4_k_mul_rows2_x16_fusion",
            512, 2048, 8, 1, 4, 2.0e-3f, true);
        {
            scoped_env_var disable_rows2_x16("GGML_HRX_DISABLE_PACKED_Q4_K_MUL_ROWS2_X16", "1");
            run_mul_mat_id_q4_k_mul_fusion_case(
                backend.get(), "mul_mat_id_q4_k_mul_packed_fusion",
                512, 2048, 8, 1, 4, 2.0e-3f, true);
        }
        {
            scoped_env_var force_q8_1("GGML_HRX_Q8_1_MMVQ", "all");
            run_mul_mat_id_q4_k_mul_fusion_case(
                backend.get(), "mul_mat_id_q4_k_mul_q8_1_fusion",
                256, 8, 2, 3, 4, 1.0e-1f, false);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(backend.get(), "mul_mat_id_q4_k_swiglu_fusion");
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            scoped_env_var disable_grouped("GGML_HRX_DISABLE_Q4_K_SWIGLU_GROUPED_PROMPT", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_row4_prompt",
                2048, 4, 8, 512, 4, 8.0e-3f, true);
        }
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_grouped_row2_prompt",
                2048, 2, 8, 512, 4, 8.0e-3f, true);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(
            backend.get(), "mul_mat_id_q4_k_swiglu_grouped_q8_1_prompt",
            2048, 16, 8, 512, 4, 2.0e-1f, true);
        {
            scoped_env_var disable_q8_1("GGML_HRX_DISABLE_Q8_1_MMVQ", "1");
            run_mul_mat_id_q4_k_swiglu_fusion_case(
                backend.get(), "mul_mat_id_q4_k_swiglu_grouped_q8_1_disabled_prompt",
                2048, 16, 8, 512, 4, 8.0e-3f, true);
        }
        run_mul_mat_id_q4_k_swiglu_fusion_case(
            backend.get(), "mul_mat_id_q4_k_swiglu_packed_fusion",
            2048, 512, 8, 1, 4, 8.0e-3f, true);
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
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, false, false, "flash_attn_ext_f16_prefill_direct");
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, true, false, "flash_attn_ext_f16_prefill_wmma16");
        run_flash_attn_ext_prefill_f16_case(
            backend.get(), dev, true, true, "flash_attn_ext_f16_prefill_tile8_softcap", 4.0f);
        run_soft_max_case(backend.get(), 1, false);
        run_soft_max_case(backend.get(), 257, false);
        run_soft_max_case(backend.get(), 257, true);
        {
            scoped_env_var enable_prompt_topk("GGML_HRX_ENABLE_PROMPT_TOPK_MOE", "1");
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "baseline");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 16, 2, 4, false, "topk_moe_baseline");
        }
        {
            scoped_env_var enable_prompt_topk("GGML_HRX_ENABLE_PROMPT_TOPK_MOE", "1");
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "shared4");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 32, 5, 8, true, "topk_moe_shared4_norm");
        }
        {
            scoped_env_var topk_variant("GGML_HRX_TOPK_MOE_VARIANT", "wave32");
            scoped_env_var disable_argsort("GGML_HRX_DISABLE_ARGSORT", "1");
            run_topk_moe_case(backend.get(), 256, 1, 32, true, "topk_moe_wave32_norm");
        }
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
