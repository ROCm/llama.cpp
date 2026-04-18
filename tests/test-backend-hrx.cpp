#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>

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
        /* .mem_size   = */ 128 * ggml_tensor_overhead() + ggml_graph_overhead_custom(16, false),
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
    if (!env_enabled("GGML_HRX_DISABLE_FAST_APPROX_PROMPT")) {
        run_unary_case(backend.get(), GGML_UNARY_OP_SILU, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SIGMOID, 257);
        run_unary_case(backend.get(), GGML_UNARY_OP_SOFTPLUS, 257);
        run_swiglu_case(backend.get(), 257);
    }
    run_clamp_case(backend.get(), 1);
    run_clamp_case(backend.get(), 256);
    run_clamp_case(backend.get(), 257);

    ggml_backend_synchronize(backend.get());
    return 0;
}
