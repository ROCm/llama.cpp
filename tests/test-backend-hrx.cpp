#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-hrx.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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

    ggml_backend_synchronize(backend.get());
    return 0;
}
