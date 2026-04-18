#include "ggml-hrx.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include "kernels/hrx_kernel_catalog.h"
#include "hrx_runtime.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace {

static constexpr size_t GGML_HRX_ALIGNMENT = 256;
static constexpr uintptr_t GGML_HRX_FAKE_PTR_BASE = 0x1000;
static constexpr size_t GGML_HRX_STAGING_ARENA_DEFAULT_SIZE = 8 * 1024 * 1024;
static constexpr uint64_t GGML_HRX_DEFAULT_NODES_PER_SUBMIT = 100;
static constexpr uint64_t GGML_HRX_DEFAULT_MAX_MUL_MAT_BYTES_PER_SUBMIT = 100ull * 1000ull * 1000ull;

struct ggml_backend_hrx_staging_arena {
    hrx_stream_t stream = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * mapped = nullptr;
    size_t capacity = 0;
    size_t offset = 0;
    std::vector<hrx_buffer_t> retired_buffers;
};

enum class ggml_backend_hrx_provider_kind {
    none,
    hsaco,
};

enum class ggml_backend_hrx_topk_moe_variant {
    auto_select,
    baseline,
    shared4,
    wave32,
};

struct ggml_backend_hrx_op_provider {
    ggml_backend_hrx_provider_kind kind = ggml_backend_hrx_provider_kind::none;
    hrx_executable_t executable = nullptr;
    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};

    ggml_backend_hrx_op_provider() = default;
    ggml_backend_hrx_op_provider(const ggml_backend_hrx_op_provider &) = delete;
    ggml_backend_hrx_op_provider & operator=(const ggml_backend_hrx_op_provider &) = delete;

    void reset() {
        if (executable) {
            hrx_executable_release(executable);
        }
        kind = ggml_backend_hrx_provider_kind::none;
        executable = nullptr;
        export_ordinal = 0;
        export_info = {};
    }

    ~ggml_backend_hrx_op_provider() {
        reset();
    }
};

struct ggml_backend_hrx_elementwise_constants {
    int64_t n;
};

static_assert(sizeof(ggml_backend_hrx_elementwise_constants) == 8);

struct ggml_backend_hrx_rms_norm_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

static_assert(sizeof(ggml_backend_hrx_rms_norm_constants) == 88);

struct ggml_backend_hrx_rms_norm_mul_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t weight_ne0;
    int64_t weight_ne1;
    int64_t weight_ne2;
    int64_t weight_ne3;
    int64_t weight_nb1;
    int64_t weight_nb2;
    int64_t weight_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

static_assert(sizeof(ggml_backend_hrx_rms_norm_mul_constants) == 144);

struct ggml_backend_hrx_rms_norm_mul_rope_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t weight_ne0;
    int64_t weight_ne1;
    int64_t weight_ne2;
    int64_t weight_ne3;
    int64_t weight_nb1;
    int64_t weight_nb2;
    int64_t weight_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad0;
    int32_t n_dims;
    int32_t mode;
    int32_t section0;
    int32_t section1;
    int32_t section2;
    int32_t section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad1;
};

static_assert(sizeof(ggml_backend_hrx_rms_norm_mul_rope_constants) == 184);

struct ggml_backend_hrx_rms_norm_mul_rope_set_rows_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t weight_ne0;
    int64_t weight_ne1;
    int64_t weight_ne2;
    int64_t weight_ne3;
    int64_t weight_nb1;
    int64_t weight_nb2;
    int64_t weight_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad0;
    int32_t n_dims;
    int32_t mode;
    int32_t section0;
    int32_t section1;
    int32_t section2;
    int32_t section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad1;
    int64_t set_rows_ne1;
    int32_t set_rows_ne11;
    int32_t set_rows_ne12;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t set_rows_dst_nb1;
    int64_t set_rows_dst_nb2;
    int64_t set_rows_dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_rms_norm_mul_rope_set_rows_constants) == 248);

struct ggml_backend_hrx_row_reduce_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    float eps;
    int32_t _pad;
};

static_assert(sizeof(ggml_backend_hrx_row_reduce_constants) == 88);

struct ggml_backend_hrx_broadcast_constants {
    int64_t ne0;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src1_ne0;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_broadcast_constants) == 112);

struct ggml_backend_hrx_add8_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t src0_nb1;
    int64_t src1_nb1;
    int64_t src2_nb1;
    int64_t src3_nb1;
    int64_t src4_nb1;
    int64_t src5_nb1;
    int64_t src6_nb1;
    int64_t src7_nb1;
    int64_t dst_nb1;
};

static_assert(sizeof(ggml_backend_hrx_add8_constants) == 88);

struct ggml_backend_hrx_mul_sum8_constants {
    int64_t rows;
    int64_t n_tokens;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t scale_nb1;
    int64_t scale_nb2;
    int64_t dst_nb1;
};

static_assert(sizeof(ggml_backend_hrx_mul_sum8_constants) == 56);

struct ggml_backend_hrx_mul_add_add_broadcast_constants {
    int64_t ne0;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src1_ne0;
    int64_t src2_ne0;
    int64_t src3_ne0;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t src2_nb1;
    int64_t src2_nb2;
    int64_t src2_nb3;
    int64_t src3_nb1;
    int64_t src3_nb2;
    int64_t src3_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_mul_add_add_broadcast_constants) == 176);

struct ggml_backend_hrx_scale_constants {
    int64_t n;
    float scale;
    float bias;
};

static_assert(sizeof(ggml_backend_hrx_scale_constants) == 16);

struct ggml_backend_hrx_clamp_constants {
    int64_t n;
    float min_value;
    float max_value;
};

static_assert(sizeof(ggml_backend_hrx_clamp_constants) == 16);

struct ggml_backend_hrx_set_rows_constants {
    int64_t nc;
    int64_t nr;
    int64_t ne02;
    int64_t ne03;
    int64_t ne1;
    int64_t ne11;
    int64_t ne12;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_set_rows_constants) == 128);

struct ggml_backend_hrx_get_rows_f32_constants {
    int64_t nc;
    int64_t nr;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    int64_t ne10;
    int64_t ne11;
};

static_assert(sizeof(ggml_backend_hrx_get_rows_f32_constants) == 104);

struct ggml_backend_hrx_mul_mat_vec_constants {
    int64_t k;
    int64_t rows;
    int64_t cols;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_vec_constants) == 24);

struct ggml_backend_hrx_mul_mat_vec_bf16_set_rows_constants {
    int64_t k;
    int64_t rows;
    int64_t set_rows_ne1;
    int64_t idx_nb0;
    int64_t dst_nb1;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_vec_bf16_set_rows_constants) == 40);

struct ggml_backend_hrx_quantize_q8_1_constants {
    int64_t ne00;
    int64_t s01;
    int64_t s02;
    int64_t s03;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
};

static_assert(sizeof(ggml_backend_hrx_quantize_q8_1_constants) == 56);

struct ggml_backend_hrx_mul_mat_vec_batched_constants {
    int64_t k;
    int64_t rows;
    int64_t cols;
    int64_t dst_ne2;
    int64_t dst_ne3;
    int64_t src0_ne2;
    int64_t src0_ne3;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src0_nb3;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t src1_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_vec_batched_constants) == 128);

struct ggml_backend_hrx_mul_mat_id_q4_k_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_constants) == 104);

struct ggml_backend_hrx_clear_u32_constants {
    int64_t n;
};

static_assert(sizeof(ggml_backend_hrx_clear_u32_constants) == 8);

struct ggml_backend_hrx_compact_moe_routes_constants {
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t route_capacity;
    int64_t ids_nb0;
    int64_t ids_nb1;
};

static_assert(sizeof(ggml_backend_hrx_compact_moe_routes_constants) == 48);

struct ggml_backend_hrx_mul_mat_id_q4_k_grouped_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t route_capacity;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_grouped_constants) == 96);

struct ggml_backend_hrx_mul_mat_id_q4_k_q8_1_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t src1_cols;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_q8_1_constants) == 96);

struct ggml_backend_hrx_mul_mat_id_q4_k_mul_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t scale_nb1;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_mul_constants) == 112);

struct ggml_backend_hrx_mul_mat_id_q4_k_swiglu_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t gate_nb1;
    int64_t gate_nb2;
    int64_t up_nb1;
    int64_t up_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_swiglu_constants) == 120);

struct ggml_backend_hrx_mul_mat_id_q4_k_swiglu_grouped_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t route_capacity;
    int64_t gate_nb1;
    int64_t gate_nb2;
    int64_t up_nb1;
    int64_t up_nb2;
    int64_t src1_nb1;
    int64_t src1_nb2;
    int64_t dst_nb1;
    int64_t dst_nb2;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_swiglu_grouped_constants) == 112);

struct ggml_backend_hrx_mul_mat_id_q4_k_mul_q8_1_constants {
    int64_t k;
    int64_t rows;
    int64_t n_ids;
    int64_t n_tokens;
    int64_t n_experts;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t ids_nb0;
    int64_t ids_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t src1_cols;
    int64_t scale_nb1;
};

static_assert(sizeof(ggml_backend_hrx_mul_mat_id_q4_k_mul_q8_1_constants) == 104);

struct ggml_backend_hrx_flash_attn_ext_f32_decode_constants {
    int64_t D;
    int64_t KV;
    int64_t N;
    int64_t H;
    int64_t H_KV;
    int64_t S;
    int64_t q_nb1;
    int64_t q_nb2;
    int64_t q_nb3;
    int64_t k_nb1;
    int64_t k_nb2;
    int64_t k_nb3;
    int64_t v_nb1;
    int64_t v_nb2;
    int64_t v_nb3;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int64_t dst_nb3;
    int64_t mask_nb0;
    int64_t mask_nb1;
    int64_t mask_nb3;
    float scale;
    int32_t has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int32_t n_head_log2;
    int32_t has_sinks;
};

static_assert(sizeof(ggml_backend_hrx_flash_attn_ext_f32_decode_constants) == 200);

struct ggml_backend_hrx_concat_f32_constants {
    int64_t ne0;
    int64_t ne1;
    int64_t src0_ne0;
    int64_t src0_nb0;
    int64_t src0_nb1;
    int64_t src1_nb0;
    int64_t src1_nb1;
    int64_t dst_nb0;
    int64_t dst_nb1;
};

static_assert(sizeof(ggml_backend_hrx_concat_f32_constants) == 72);

struct ggml_backend_hrx_copy_strided_f32_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne1;
    int64_t ne2;
    int64_t src_nb1;
    int64_t src_nb2;
    int64_t src_nb3;
    int64_t row_size;
};

static_assert(sizeof(ggml_backend_hrx_copy_strided_f32_constants) == 64);

struct ggml_backend_hrx_copy_f32_f16_constants {
    int64_t n;
};

static_assert(sizeof(ggml_backend_hrx_copy_f32_f16_constants) == 8);

struct ggml_backend_hrx_soft_max_f32_constants {
    int64_t ncols;
    int64_t nrows;
    int64_t ne01;
    int64_t ne02;
    int64_t mask_nb1;
    int64_t mask_nb2;
    int64_t mask_nb3;
    int64_t mask_ne1;
    int64_t mask_ne2;
    int64_t mask_ne3;
    float scale;
    int32_t _pad;
};

static_assert(sizeof(ggml_backend_hrx_soft_max_f32_constants) == 88);

struct ggml_backend_hrx_argsort_f32_constants {
    int64_t ncols;
    int64_t nrows;
    int32_t order;
    int32_t ncols_pad;
};

static_assert(sizeof(ggml_backend_hrx_argsort_f32_constants) == 24);

struct ggml_backend_hrx_topk_moe_f32_constants {
    int64_t n_experts;
    int64_t n_rows;
    int64_t n_expert_used;
    int64_t logits_nb1;
    int64_t weights_nb1;
    int64_t weights_nb_k;
    int64_t ids_nb1;
    int64_t ids_nb_k;
    float scale;
    float clamp_min;
    float clamp_max;
    int32_t with_norm;
};

static_assert(sizeof(ggml_backend_hrx_topk_moe_f32_constants) == 80);

struct ggml_backend_hrx_rope_f32_constants {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t nrows;
    int64_t src_s1;
    int64_t src_s2;
    int64_t src_s3;
    int64_t dst_s1;
    int64_t dst_s2;
    int64_t dst_s3;
    int32_t n_dims;
    int32_t mode;
    int32_t section0;
    int32_t section1;
    int32_t section2;
    int32_t section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad;
};

static_assert(sizeof(ggml_backend_hrx_rope_f32_constants) == 120);

struct ggml_backend_hrx_rope_set_rows_f32_f16_constants {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t nrows;
    int64_t src_s1;
    int64_t src_s2;
    int64_t src_s3;
    int64_t dst_s1;
    int64_t dst_s2;
    int64_t dst_s3;
    int32_t n_dims;
    int32_t mode;
    int32_t section0;
    int32_t section1;
    int32_t section2;
    int32_t section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad;
    int64_t set_rows_ne1;
    int64_t set_rows_ne11;
    int64_t set_rows_ne12;
    int64_t idx_nb0;
    int64_t idx_nb1;
    int64_t idx_nb2;
    int64_t set_rows_dst_nb1;
    int64_t set_rows_dst_nb2;
    int64_t set_rows_dst_nb3;
};

static_assert(sizeof(ggml_backend_hrx_rope_set_rows_f32_f16_constants) == 192);

struct ggml_backend_hrx_ssm_conv_constants {
    int64_t d_conv;
    int64_t conv_width;
    int64_t d_inner;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t src0_nb1;
    int64_t src0_nb2;
    int64_t weight_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int32_t apply_silu;
    int32_t pad;
};

static_assert(sizeof(ggml_backend_hrx_ssm_conv_constants) == 88);

struct ggml_backend_hrx_ssm_conv_update_constants {
    int64_t d_conv;
    int64_t conv_state_width;
    int64_t d_inner;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t state_nb0;
    int64_t state_nb1;
    int64_t state_nb2;
    int64_t input_nb0;
    int64_t input_nb1;
    int64_t weight_nb1;
    int64_t dst_nb1;
    int64_t dst_nb2;
    int32_t apply_silu;
    int32_t pad;
};

static_assert(sizeof(ggml_backend_hrx_ssm_conv_update_constants) == 112);

struct ggml_backend_hrx_gated_delta_net_constants {
    int64_t S_v;
    int64_t H;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t neq1;
    int64_t nek1;
    int64_t rq3;
    int64_t rk3;
    int64_t q_nb1;
    int64_t q_nb2;
    int64_t q_nb3;
    int64_t k_nb1;
    int64_t k_nb2;
    int64_t k_nb3;
    int64_t v_nb1;
    int64_t v_nb2;
    int64_t v_nb3;
    int64_t g_ne0;
    int64_t g_nb1;
    int64_t g_nb2;
    int64_t g_nb3;
    int64_t beta_nb1;
    int64_t beta_nb2;
    int64_t beta_nb3;
    int64_t state_dst_offset;
    float scale;
    int32_t _pad;
};

static_assert(sizeof(ggml_backend_hrx_gated_delta_net_constants) == 208);

struct ggml_backend_hrx_device_context {
    hrx_device_t device = nullptr;
    hrx_stream_t active_stream = nullptr;
    hrx_stream_t transfer_stream = nullptr;
    std::mutex streams_mutex;
    std::vector<hrx_stream_t> live_streams;
    std::vector<ggml_backend_hrx_staging_arena> staging_arenas;
    std::string name;
    std::string description;
    std::string architecture;
    size_t memory_total = 0;
    ggml_backend_hrx_op_provider rms_norm_provider;
    ggml_backend_hrx_op_provider rms_norm_mul_provider;
    ggml_backend_hrx_op_provider rms_norm_mul_wg128_provider;
    ggml_backend_hrx_op_provider rms_norm_mul_rope_f32_provider;
    ggml_backend_hrx_op_provider rms_norm_mul_rope_set_rows_f32_f16_provider;
    ggml_backend_hrx_op_provider add_provider;
    ggml_backend_hrx_op_provider mul_provider;
    ggml_backend_hrx_op_provider add_broadcast_provider;
    ggml_backend_hrx_op_provider mul_broadcast_provider;
    ggml_backend_hrx_op_provider div_broadcast_provider;
    ggml_backend_hrx_op_provider add8_provider;
    ggml_backend_hrx_op_provider mul_sum8_provider;
    ggml_backend_hrx_op_provider mul_add_add_broadcast_provider;
    ggml_backend_hrx_op_provider scale_provider;
    ggml_backend_hrx_op_provider set_rows_f32_provider;
    ggml_backend_hrx_op_provider set_rows_f16_provider;
    ggml_backend_hrx_op_provider set_rows_q8_0_provider;
    ggml_backend_hrx_op_provider set_rows_q4_0_provider;
    ggml_backend_hrx_op_provider silu_provider;
    ggml_backend_hrx_op_provider sigmoid_provider;
    ggml_backend_hrx_op_provider softplus_provider;
    ggml_backend_hrx_op_provider swiglu_provider;
    ggml_backend_hrx_op_provider sum_rows_provider;
    ggml_backend_hrx_op_provider l2_norm_provider;
    ggml_backend_hrx_op_provider l2_norm_wg128_provider;
    ggml_backend_hrx_op_provider clamp_provider;
    ggml_backend_hrx_op_provider get_rows_f32_provider;
    ggml_backend_hrx_op_provider get_rows_f32_nr1_provider;
    ggml_backend_hrx_op_provider get_rows_q5_k_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_wg128_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_rows2_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_rows2_cols1_wg32_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_rows4_k512_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_rows4_k2048_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_cols4_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_cols16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_cols32_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_rows2_cols16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_wmma16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_wg128_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_rows2_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_cols4_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_cols16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_rows2_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_swiglu_wmma16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_bf16_set_rows_f16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_batched_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_batched_cols1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_batched_cols4_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_batched_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f16_batched_cols16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_cols1_ne2_1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_cols16_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_f32_batched_rows2_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_row4_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_wg64_provider;
    ggml_backend_hrx_op_provider clear_u32_provider;
    ggml_backend_hrx_op_provider compact_moe_routes_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_grouped_row2_route8_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_q8_1_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_mul_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_mul_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_mul_packed_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_mul_rows2_x16_wg32_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_row4_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_swiglu_packed_wg64_provider;
    ggml_backend_hrx_op_provider mul_mat_id_q4_k_mul_q8_1_provider;
    ggml_backend_hrx_op_provider quantize_q8_1_provider;
    ggml_backend_hrx_op_provider quantize_q8_1_x4_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q4_k_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q4_k_q8_1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_q8_1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q6_k_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q6_k_q8_1_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_add_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_add_cols8_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_add_rows4_cols4_provider;
    ggml_backend_hrx_op_provider mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_f16_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_f16_prefill_tile_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_f16_prefill_wmma_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_f16_prefill_direct_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_bf16_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_f32_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_q4_0_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_q8_0_provider;
    ggml_backend_hrx_op_provider flash_attn_ext_q8_0_q4_0_provider;
    ggml_backend_hrx_op_provider concat_f32_provider;
    ggml_backend_hrx_op_provider copy_strided_f32_provider;
    ggml_backend_hrx_op_provider copy_f32_f16_provider;
    ggml_backend_hrx_op_provider soft_max_f32_provider;
    ggml_backend_hrx_op_provider soft_max_f32_mask_provider;
    ggml_backend_hrx_op_provider argsort_f32_provider;
    ggml_backend_hrx_op_provider topk_moe_f32_provider;
    ggml_backend_hrx_op_provider topk_moe_f32_shared4_provider;
    ggml_backend_hrx_op_provider topk_moe_f32_wave32_provider;
    ggml_backend_hrx_op_provider rope_f32_provider;
    ggml_backend_hrx_op_provider rope_set_rows_f32_f16_provider;
    ggml_backend_hrx_op_provider ssm_conv_provider;
    ggml_backend_hrx_op_provider ssm_conv_update_provider;
    ggml_backend_hrx_op_provider gated_delta_net_provider;
};

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream);

static void ggml_backend_hrx_reset_providers(ggml_backend_hrx_device_context * device_context) {
    device_context->rms_norm_provider.reset();
    device_context->rms_norm_mul_provider.reset();
    device_context->rms_norm_mul_wg128_provider.reset();
    device_context->rms_norm_mul_rope_f32_provider.reset();
    device_context->rms_norm_mul_rope_set_rows_f32_f16_provider.reset();
    device_context->add_provider.reset();
    device_context->mul_provider.reset();
    device_context->add_broadcast_provider.reset();
    device_context->mul_broadcast_provider.reset();
    device_context->div_broadcast_provider.reset();
    device_context->add8_provider.reset();
    device_context->mul_sum8_provider.reset();
    device_context->mul_add_add_broadcast_provider.reset();
    device_context->scale_provider.reset();
    device_context->set_rows_f32_provider.reset();
    device_context->set_rows_f16_provider.reset();
    device_context->set_rows_q8_0_provider.reset();
    device_context->set_rows_q4_0_provider.reset();
    device_context->silu_provider.reset();
    device_context->sigmoid_provider.reset();
    device_context->softplus_provider.reset();
    device_context->swiglu_provider.reset();
    device_context->sum_rows_provider.reset();
    device_context->l2_norm_provider.reset();
    device_context->l2_norm_wg128_provider.reset();
    device_context->clamp_provider.reset();
    device_context->get_rows_f32_provider.reset();
    device_context->get_rows_f32_nr1_provider.reset();
    device_context->get_rows_q5_k_provider.reset();
    device_context->mul_mat_vec_bf16_provider.reset();
    device_context->mul_mat_vec_bf16_wg128_provider.reset();
    device_context->mul_mat_vec_bf16_wg64_provider.reset();
    device_context->mul_mat_vec_bf16_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_rows2_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_rows2_cols1_wg32_provider.reset();
    device_context->mul_mat_vec_bf16_rows4_k512_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_rows4_k2048_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_cols4_provider.reset();
    device_context->mul_mat_vec_bf16_cols8_provider.reset();
    device_context->mul_mat_vec_bf16_cols16_provider.reset();
    device_context->mul_mat_vec_bf16_cols32_provider.reset();
    device_context->mul_mat_vec_bf16_rows2_cols16_provider.reset();
    device_context->mul_mat_vec_bf16_wmma16_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_wg128_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_wg64_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_rows2_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_cols4_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_cols8_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_cols16_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider.reset();
    device_context->mul_mat_vec_bf16_swiglu_wmma16_provider.reset();
    device_context->mul_mat_vec_bf16_set_rows_f16_provider.reset();
    device_context->mul_mat_vec_f16_provider.reset();
    device_context->mul_mat_vec_f16_batched_provider.reset();
    device_context->mul_mat_vec_f16_batched_cols1_provider.reset();
    device_context->mul_mat_vec_f16_batched_cols4_provider.reset();
    device_context->mul_mat_vec_f16_batched_cols8_provider.reset();
    device_context->mul_mat_vec_f16_batched_cols16_provider.reset();
    device_context->mul_mat_vec_f32_provider.reset();
    device_context->mul_mat_vec_f32_batched_provider.reset();
    device_context->mul_mat_vec_f32_batched_cols1_ne2_1_provider.reset();
    device_context->mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_provider.reset();
    device_context->mul_mat_vec_f32_batched_cols8_provider.reset();
    device_context->mul_mat_vec_f32_batched_cols16_provider.reset();
    device_context->mul_mat_vec_f32_batched_rows2_cols8_provider.reset();
    device_context->mul_mat_id_q4_k_provider.reset();
    device_context->mul_mat_id_q4_k_row4_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_wg64_provider.reset();
    device_context->clear_u32_provider.reset();
    device_context->compact_moe_routes_provider.reset();
    device_context->mul_mat_id_q4_k_grouped_row2_route8_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_q8_1_provider.reset();
    device_context->mul_mat_id_q4_k_mul_provider.reset();
    device_context->mul_mat_id_q4_k_mul_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_mul_packed_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_mul_rows2_x16_wg32_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_row4_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_swiglu_packed_wg64_provider.reset();
    device_context->mul_mat_id_q4_k_mul_q8_1_provider.reset();
    device_context->quantize_q8_1_provider.reset();
    device_context->quantize_q8_1_x4_provider.reset();
    device_context->mul_mat_vec_q4_k_provider.reset();
    device_context->mul_mat_vec_q4_k_q8_1_provider.reset();
    device_context->mul_mat_vec_q5_k_provider.reset();
    device_context->mul_mat_vec_q5_k_q8_1_provider.reset();
    device_context->mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_provider.reset();
    device_context->mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_provider.reset();
    device_context->mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_provider.reset();
    device_context->mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_provider.reset();
    device_context->mul_mat_vec_q6_k_provider.reset();
    device_context->mul_mat_vec_q6_k_q8_1_provider.reset();
    device_context->mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_provider.reset();
    device_context->mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_provider.reset();
    device_context->mul_mat_vec_q8_0_provider.reset();
    device_context->mul_mat_vec_q8_0_cols8_provider.reset();
    device_context->mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_provider.reset();
    device_context->mul_mat_vec_q8_0_add_provider.reset();
    device_context->mul_mat_vec_q8_0_add_cols8_provider.reset();
    device_context->mul_mat_vec_q8_0_add_rows4_cols4_provider.reset();
    device_context->mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider.reset();
    device_context->flash_attn_ext_f16_provider.reset();
    device_context->flash_attn_ext_f16_prefill_tile_provider.reset();
    device_context->flash_attn_ext_f16_prefill_wmma_provider.reset();
    device_context->flash_attn_ext_f16_prefill_direct_provider.reset();
    device_context->flash_attn_ext_bf16_provider.reset();
    device_context->flash_attn_ext_f32_provider.reset();
    device_context->flash_attn_ext_q4_0_provider.reset();
    device_context->flash_attn_ext_q8_0_provider.reset();
    device_context->flash_attn_ext_q8_0_q4_0_provider.reset();
    device_context->concat_f32_provider.reset();
    device_context->copy_strided_f32_provider.reset();
    device_context->copy_f32_f16_provider.reset();
    device_context->soft_max_f32_provider.reset();
    device_context->soft_max_f32_mask_provider.reset();
    device_context->argsort_f32_provider.reset();
    device_context->topk_moe_f32_provider.reset();
    device_context->topk_moe_f32_shared4_provider.reset();
    device_context->topk_moe_f32_wave32_provider.reset();
    device_context->rope_f32_provider.reset();
    device_context->rope_set_rows_f32_f16_provider.reset();
    device_context->ssm_conv_provider.reset();
    device_context->ssm_conv_update_provider.reset();
    device_context->gated_delta_net_provider.reset();
}

struct ggml_backend_hrx_reg_context {
    bool gpu_initialized = false;
    std::vector<std::unique_ptr<ggml_backend_hrx_device_context>> device_contexts;
    std::vector<ggml_backend_device> devices;

    ~ggml_backend_hrx_reg_context() {
        for (auto & device_context : device_contexts) {
            if (device_context) {
                ggml_backend_hrx_reset_providers(device_context.get());
            }
            if (device_context && device_context->transfer_stream) {
                hrx_status_t status = hrx_stream_synchronize(device_context->transfer_stream);
                if (!hrx_status_is_ok(status)) {
                    hrx_status_ignore(status);
                }
                ggml_backend_hrx_unregister_stream(device_context.get(), device_context->transfer_stream);
                hrx_stream_release(device_context->transfer_stream);
                device_context->transfer_stream = nullptr;
            }
            if (device_context && device_context->device) {
                hrx_device_release(device_context->device);
                device_context->device = nullptr;
            }
        }
        if (gpu_initialized) {
            hrx_status_t status = hrx_gpu_shutdown();
            if (!hrx_status_is_ok(status)) {
                hrx_status_ignore(status);
            }
        }
    }
};

struct ggml_backend_hrx_buffer_type_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    std::string name;
    hrx_buffer_params_t params = {};
};

struct ggml_backend_hrx_buffer_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_buffer_t buffer = nullptr;
    uint8_t * base = nullptr;
};

enum class ggml_backend_hrx_scratch_state {
    available,
    in_use,
    retired,
};

struct ggml_backend_hrx_scratch_buffer {
    hrx_buffer_t buffer = nullptr;
    size_t size = 0;
    ggml_backend_hrx_scratch_state state = ggml_backend_hrx_scratch_state::available;
};

struct ggml_backend_hrx_context {
    ggml_backend_hrx_device_context * device_context = nullptr;
    hrx_stream_t stream = nullptr;
    std::string name;
    std::vector<ggml_backend_hrx_scratch_buffer> scratch_buffers;
    uint64_t last_total_mul_mat_bytes = 0;
    uint64_t submitted_nodes = 0;
    uint64_t mul_mat_bytes = 0;
    uint64_t total_mul_mat_bytes = 0;
    uint64_t mul_mat_bytes_per_submit = 0;
    uint64_t submit_count = 0;
    uint64_t submit_flush_count = 0;
    const ggml_tensor * last_submitted_node = nullptr;
};

static thread_local ggml_backend_hrx_context * g_hrx_active_graph_context = nullptr;
static thread_local const ggml_tensor * g_hrx_active_graph_node = nullptr;

static bool ggml_backend_hrx_log_status(hrx_status_t status, const char * expr, const char * file, int line) {
    if (hrx_status_is_ok(status)) {
        return true;
    }

    char * message = nullptr;
    size_t length = 0;
    hrx_status_to_string(status, &message, &length);
    GGML_LOG_ERROR("%s:%d: %s failed: %s\n", file, line, expr, message ? message : "unknown HRX error");
    hrx_status_free_message(message);
    hrx_status_ignore(status);
    return false;
}

#define GGML_HRX_CHECK(expr) ggml_backend_hrx_log_status((expr), #expr, __FILE__, __LINE__)

static uint64_t ggml_backend_hrx_u64_from_env(const char * name, uint64_t default_value) {
    const char * value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return default_value;
    }
    errno = 0;
    char * end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0') {
        GGML_LOG_WARN("%s: ignoring invalid %s=%s\n", __func__, name, value);
        return default_value;
    }
    return static_cast<uint64_t>(parsed);
}

static bool ggml_backend_hrx_env_enabled(const char * name) {
    const char * value = std::getenv(name);
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static uint64_t ggml_backend_hrx_nodes_per_submit() {
    return ggml_backend_hrx_u64_from_env("GGML_HRX_NODES_PER_SUBMIT", GGML_HRX_DEFAULT_NODES_PER_SUBMIT);
}

static uint64_t ggml_backend_hrx_max_mul_mat_bytes_per_submit() {
    return ggml_backend_hrx_u64_from_env(
        "GGML_HRX_MAX_MUL_MAT_BYTES_PER_SUBMIT", GGML_HRX_DEFAULT_MAX_MUL_MAT_BYTES_PER_SUBMIT);
}

static uint64_t ggml_backend_hrx_node_mul_mat_bytes(const ggml_tensor * node) {
    if (!node || !node->src[0]) {
        return 0;
    }
    if (node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID) {
        return 0;
    }
    return static_cast<uint64_t>(ggml_nbytes(node->src[0]));
}

static void ggml_backend_hrx_begin_submit_batch(ggml_backend_hrx_context * context) {
    if (!context) {
        return;
    }
    const uint64_t max_bytes = ggml_backend_hrx_max_mul_mat_bytes_per_submit();
    const uint64_t last_scaled = context->last_total_mul_mat_bytes / 40u;
    context->submitted_nodes = 0;
    context->mul_mat_bytes = 0;
    context->total_mul_mat_bytes = 0;
    context->mul_mat_bytes_per_submit = std::min(max_bytes, last_scaled);
    context->submit_count = 0;
    context->submit_flush_count = 0;
    context->last_submitted_node = nullptr;
}

static hrx_status_t ggml_backend_hrx_maybe_submit_batch_after_dispatch(hrx_stream_t stream) {
    ggml_backend_hrx_context * context = g_hrx_active_graph_context;
    if (!context || stream != context->stream || ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_SUBMIT_BATCHING")) {
        return hrx_ok_status();
    }

    const ggml_tensor * node = g_hrx_active_graph_node;
    if (node && node != context->last_submitted_node) {
        const uint64_t matmul_bytes = ggml_backend_hrx_node_mul_mat_bytes(node);
        context->last_submitted_node = node;
        context->submitted_nodes++;
        context->mul_mat_bytes += matmul_bytes;
        context->total_mul_mat_bytes += matmul_bytes;
    }

    const uint64_t nodes_per_submit = ggml_backend_hrx_nodes_per_submit();
    const bool node_threshold = nodes_per_submit != 0 && context->submitted_nodes >= nodes_per_submit;
    const bool byte_threshold =
        context->mul_mat_bytes_per_submit != 0 && context->mul_mat_bytes >= context->mul_mat_bytes_per_submit;
    if (!node_threshold && !byte_threshold) {
        return hrx_ok_status();
    }

    if (ggml_backend_hrx_env_enabled("GGML_HRX_TRACE_SUBMIT_BATCHING")) {
        GGML_LOG_DEBUG(
            "%s: submit nodes=%" PRIu64 " mul_mat_bytes=%" PRIu64
            " mul_mat_bytes_per_submit=%" PRIu64 " submit_count=%" PRIu64 "\n",
            __func__,
            context->submitted_nodes,
            context->mul_mat_bytes,
            context->mul_mat_bytes_per_submit,
            context->submit_count);
    }

    hrx_status_t status = hrx_stream_flush(stream);
    if (!hrx_status_is_ok(status)) {
        return status;
    }

    context->submitted_nodes = 0;
    context->mul_mat_bytes = 0;
    if (context->submit_count < 3) {
        context->mul_mat_bytes_per_submit *= 2;
    }
    context->submit_count++;
    context->submit_flush_count++;
    return hrx_ok_status();
}

static hrx_status_t ggml_backend_hrx_stream_dispatch(
        hrx_stream_t stream,
        hrx_executable_t executable,
        uint32_t export_ordinal,
        const hrx_dispatch_config_t * config,
        const void * constants,
        size_t constants_size,
        const hrx_buffer_ref_t * bindings,
        size_t binding_count,
        uint32_t flags) {
    hrx_status_t status = hrx_stream_dispatch(
        stream,
        executable,
        export_ordinal,
        config,
        constants,
        constants_size,
        bindings,
        binding_count,
        flags);
    if (!hrx_status_is_ok(status)) {
        return status;
    }
    return ggml_backend_hrx_maybe_submit_batch_after_dispatch(stream);
}

#define hrx_stream_dispatch ggml_backend_hrx_stream_dispatch

static ggml_backend_hrx_topk_moe_variant ggml_backend_hrx_topk_moe_variant_from_env() {
    const char * value = std::getenv("GGML_HRX_TOPK_MOE_VARIANT");
    if (!value || value[0] == '\0' || std::strcmp(value, "auto") == 0) {
        return ggml_backend_hrx_topk_moe_variant::auto_select;
    }
    if (std::strcmp(value, "baseline") == 0 || std::strcmp(value, "shared1") == 0) {
        return ggml_backend_hrx_topk_moe_variant::baseline;
    }
    if (std::strcmp(value, "shared4") == 0 || std::strcmp(value, "shared") == 0) {
        return ggml_backend_hrx_topk_moe_variant::shared4;
    }
    if (std::strcmp(value, "wave32") == 0) {
        return ggml_backend_hrx_topk_moe_variant::wave32;
    }
    GGML_LOG_WARN("%s: unknown GGML_HRX_TOPK_MOE_VARIANT=%s, using auto\n", __func__, value);
    return ggml_backend_hrx_topk_moe_variant::auto_select;
}

static size_t ggml_backend_hrx_align_up(size_t value, size_t alignment) {
    GGML_ASSERT(alignment > 0);
    const size_t remainder = value % alignment;
    return remainder == 0 ? value : value + (alignment - remainder);
}

static size_t ggml_backend_hrx_staging_arena_capacity() {
    const uint64_t requested = ggml_backend_hrx_u64_from_env(
        "GGML_HRX_STAGING_ARENA_SIZE", GGML_HRX_STAGING_ARENA_DEFAULT_SIZE);
    const size_t capacity = static_cast<size_t>(std::max<uint64_t>(requested, GGML_HRX_ALIGNMENT));
    return ggml_backend_hrx_align_up(capacity, GGML_HRX_ALIGNMENT);
}

static ggml_guid_t ggml_backend_hrx_guid(void) {
    static ggml_guid guid = { 0x1c, 0x65, 0x79, 0x0a, 0x31, 0x8b, 0x4d, 0xa6, 0x9e, 0x16, 0x6f, 0x13, 0x39, 0xb2, 0xe7, 0x5c };
    return &guid;
}

static ggml_backend_hrx_device_context * ggml_backend_hrx_get_device_context(ggml_backend_dev_t dev) {
    return static_cast<ggml_backend_hrx_device_context *>(dev->context);
}

static ggml_backend_hrx_buffer_type_context * ggml_backend_hrx_get_buft_context(ggml_backend_buffer_type_t buft) {
    return static_cast<ggml_backend_hrx_buffer_type_context *>(buft->context);
}

static ggml_backend_hrx_buffer_context * ggml_backend_hrx_get_buffer_context(ggml_backend_buffer_t buffer) {
    return static_cast<ggml_backend_hrx_buffer_context *>(buffer->context);
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer);

static size_t ggml_backend_hrx_tensor_offset(const ggml_backend_hrx_buffer_context * context, const ggml_tensor * tensor) {
    return static_cast<size_t>(static_cast<const uint8_t *>(tensor->data) - context->base);
}

static bool ggml_backend_hrx_tensor_buffer_ref(
        const ggml_tensor * tensor, hrx_buffer_ref_t * out_ref) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (!context->buffer) {
        return false;
    }

    const size_t offset = ggml_backend_hrx_tensor_offset(context, tensor);
    const size_t length = ggml_nbytes(tensor);
    if (offset > buffer->size || length > buffer->size - offset) {
        GGML_LOG_ERROR(
            "%s: tensor %s has out-of-bounds HRX buffer ref: offset=%zu length=%zu buffer_size=%zu\n",
            __func__, tensor->name, offset, length, buffer->size);
        return false;
    }

    *out_ref = {
        /* .buffer = */ context->buffer,
        /* .offset = */ offset,
        /* .length = */ length,
    };
    return true;
}

static void ggml_backend_hrx_retire_in_use_scratch_buffers(ggml_backend_hrx_context * context) {
    for (ggml_backend_hrx_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state == ggml_backend_hrx_scratch_state::in_use) {
            scratch.state = ggml_backend_hrx_scratch_state::retired;
        }
    }
}

static void ggml_backend_hrx_recycle_scratch_buffers(ggml_backend_hrx_context * context) {
    for (ggml_backend_hrx_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state != ggml_backend_hrx_scratch_state::available) {
            scratch.state = ggml_backend_hrx_scratch_state::available;
        }
    }
}

static void ggml_backend_hrx_release_scratch_buffers(ggml_backend_hrx_context * context) {
    for (ggml_backend_hrx_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.buffer) {
            hrx_buffer_release(scratch.buffer);
            scratch.buffer = nullptr;
        }
        scratch.size = 0;
        scratch.state = ggml_backend_hrx_scratch_state::available;
    }
    context->scratch_buffers.clear();
}

static bool ggml_backend_hrx_request_scratch_buffer(
        ggml_backend_hrx_context * context,
        size_t size,
        hrx_buffer_ref_t * out_ref) {
    if (size == 0) {
        return false;
    }

    ggml_backend_hrx_retire_in_use_scratch_buffers(context);

    ggml_backend_hrx_scratch_buffer * selected = nullptr;
    for (ggml_backend_hrx_scratch_buffer & scratch : context->scratch_buffers) {
        if (scratch.state != ggml_backend_hrx_scratch_state::available || scratch.size < size) {
            continue;
        }
        if (!selected || scratch.size < selected->size) {
            selected = &scratch;
        }
    }

    if (!selected) {
        hrx_buffer_params_t params = {
            /* .type           = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access         = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage          = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        };
        hrx_buffer_t buffer = nullptr;
        if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
                hrx_device_allocator(context->device_context->device),
                params,
                size,
                &buffer))) {
            return false;
        }
        context->scratch_buffers.push_back({
            /* .buffer = */ buffer,
            /* .size   = */ size,
            /* .state  = */ ggml_backend_hrx_scratch_state::available,
        });
        selected = &context->scratch_buffers.back();
    }

    selected->state = ggml_backend_hrx_scratch_state::in_use;
    *out_ref = {
        /* .buffer = */ selected->buffer,
        /* .offset = */ 0,
        /* .length = */ size,
    };
    return true;
}

static void ggml_backend_hrx_register_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }
    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    if (std::find(device_context->live_streams.begin(), device_context->live_streams.end(), stream) ==
            device_context->live_streams.end()) {
        device_context->live_streams.push_back(stream);
    }
}

static void ggml_backend_hrx_reset_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena.retired_buffers.clear();
    arena.offset = 0;
}

static void ggml_backend_hrx_release_staging_arena_locked(ggml_backend_hrx_staging_arena & arena) {
    if (arena.buffer) {
        hrx_buffer_release(arena.buffer);
    }
    for (hrx_buffer_t buffer : arena.retired_buffers) {
        hrx_buffer_release(buffer);
    }
    arena = {};
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_find_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    for (auto & arena : device_context->staging_arenas) {
        if (arena.stream == stream) {
            return &arena;
        }
    }
    return nullptr;
}

static ggml_backend_hrx_staging_arena * ggml_backend_hrx_get_staging_arena_locked(
        ggml_backend_hrx_device_context * device_context,
        hrx_stream_t stream) {
    if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
        return arena;
    }
    device_context->staging_arenas.push_back({});
    auto & arena = device_context->staging_arenas.back();
    arena.stream = stream;
    return &arena;
}

static void ggml_backend_hrx_unregister_stream(ggml_backend_hrx_device_context * device_context, hrx_stream_t stream) {
    if (!device_context || !stream) {
        return;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    auto & streams = device_context->live_streams;
    streams.erase(std::remove(streams.begin(), streams.end(), stream), streams.end());
    auto & arenas = device_context->staging_arenas;
    auto arena_it = std::find_if(
        arenas.begin(), arenas.end(),
        [stream](const ggml_backend_hrx_staging_arena & arena) { return arena.stream == stream; });
    if (arena_it != arenas.end()) {
        ggml_backend_hrx_release_staging_arena_locked(*arena_it);
        arenas.erase(arena_it);
    }
    if (device_context->active_stream == stream) {
        device_context->active_stream = nullptr;
    }
}

static hrx_stream_t ggml_backend_hrx_retain_timeline_stream(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    hrx_stream_t stream = device_context->active_stream;
    if (!stream) {
        stream = device_context->transfer_stream;
    }
    if (stream) {
        hrx_stream_retain(stream);
    }
    return stream;
}

static bool ggml_backend_hrx_sync_streams(ggml_backend_hrx_device_context * device_context) {
    if (!device_context) {
        return true;
    }

    std::lock_guard<std::mutex> lock(device_context->streams_mutex);
    bool ok = true;
    for (hrx_stream_t stream : device_context->live_streams) {
        ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream)) && ok;
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(device_context, stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
    return ok;
}

static bool ggml_backend_hrx_prepare_stream_signal(
        hrx_stream_t stream,
        hrx_semaphore_t * semaphore,
        uint64_t * signal_value,
        hrx_semaphore_list_t * wait_list,
        hrx_semaphore_list_t * signal_list,
        hrx_semaphore_t * wait_semaphores,
        uint64_t * wait_values,
        hrx_semaphore_t * signal_semaphores,
        uint64_t * signal_values) {
    hrx_timeline_point_t position = {};
    if (!GGML_HRX_CHECK(hrx_stream_flush(stream)) ||
        !GGML_HRX_CHECK(hrx_stream_get_timeline_position(stream, &position)) ||
        !GGML_HRX_CHECK(hrx_stream_get_semaphore(stream, semaphore))) {
        return false;
    }

    *signal_value = position.value + 1;
    if (position.value > 0) {
        wait_semaphores[0] = *semaphore;
        wait_values[0] = position.value;
        *wait_list = {
            /* .semaphores = */ wait_semaphores,
            /* .values     = */ wait_values,
            /* .count      = */ 1,
        };
    } else {
        *wait_list = {};
    }

    signal_semaphores[0] = *semaphore;
    signal_values[0] = *signal_value;
    *signal_list = {
        /* .semaphores = */ signal_semaphores,
        /* .values     = */ signal_values,
        /* .count      = */ 1,
    };
    return true;
}

static bool ggml_backend_hrx_finish_stream_signal(hrx_stream_t stream, uint64_t signal_value) {
    uint64_t advanced_value = 0;
    if (!GGML_HRX_CHECK(hrx_stream_advance_timeline(stream, &advanced_value))) {
        return false;
    }
    if (advanced_value != signal_value) {
        GGML_LOG_ERROR("%s: stream timeline advanced to %" PRIu64 ", expected %" PRIu64 "\n",
                __func__, advanced_value, signal_value);
        return false;
    }
    return GGML_HRX_CHECK(hrx_stream_wait(stream));
}

static bool ggml_backend_hrx_queue_fill_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t buffer,
        size_t offset,
        size_t size,
        const void * pattern,
        size_t pattern_size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous fill\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_fill(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, buffer, offset, size, pattern, pattern_size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_queue_copy_stream_sync(
        ggml_backend_hrx_device_context * device_context,
        hrx_buffer_t src,
        size_t src_offset,
        hrx_buffer_t dst,
        size_t dst_offset,
        size_t size) {
    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream registered for synchronous copy\n", __func__);
        return false;
    }

    hrx_semaphore_t semaphore = nullptr;
    uint64_t signal_value = 0;
    hrx_semaphore_t wait_semaphores[1] = {};
    uint64_t wait_values[1] = {};
    hrx_semaphore_t signal_semaphores[1] = {};
    uint64_t signal_values[1] = {};
    hrx_semaphore_list_t wait_list = {};
    hrx_semaphore_list_t signal_list = {};
    bool ok = ggml_backend_hrx_prepare_stream_signal(
        stream, &semaphore, &signal_value, &wait_list, &signal_list,
        wait_semaphores, wait_values, signal_semaphores, signal_values);
    ok = ok && GGML_HRX_CHECK(hrx_queue_copy(
        device_context->device, 0,
        wait_list.count ? &wait_list : nullptr,
        &signal_list, src, src_offset, dst, dst_offset, size));
    ok = ok && ggml_backend_hrx_finish_stream_signal(stream, signal_value);
    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_ensure_staging_buffer_locked(
        ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_staging_arena * arena,
        size_t required_capacity) {
    if (arena->buffer && arena->capacity >= required_capacity && arena->mapped) {
        return true;
    }

    if (arena->buffer) {
        arena->retired_buffers.push_back(arena->buffer);
        arena->buffer = nullptr;
        arena->mapped = nullptr;
        arena->capacity = 0;
        arena->offset = 0;
    }

    const size_t capacity = ggml_backend_hrx_align_up(
        std::max(required_capacity, ggml_backend_hrx_staging_arena_capacity()),
        GGML_HRX_ALIGNMENT);
    hrx_buffer_params_t params = {
        /* .type = */ HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_DEVICE_VISIBLE,
        /* .access = */ HRX_MEMORY_ACCESS_ALL,
        /* .usage = */ HRX_BUFFER_USAGE_DEFAULT |
                       HRX_BUFFER_USAGE_MAPPING_SCOPED |
                       HRX_BUFFER_USAGE_MAPPING_PERSISTENT,
        /* .queue_affinity = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(device_context->device), params, capacity, &arena->buffer))) {
        return false;
    }

    void * mapped = nullptr;
    if (!GGML_HRX_CHECK(hrx_buffer_map(arena->buffer, HRX_MAP_READ | HRX_MAP_WRITE, 0, capacity, &mapped))) {
        hrx_buffer_release(arena->buffer);
        arena->buffer = nullptr;
        return false;
    }
    arena->mapped = static_cast<uint8_t *>(mapped);
    arena->capacity = capacity;
    arena->offset = 0;
    return true;
}

static bool ggml_backend_hrx_stage_and_copy_tensor(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        const void * data,
        size_t buffer_offset,
        size_t buffer_size,
        size_t size) {
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: upload for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor upload\n", __func__);
        return false;
    }

    std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
    auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
    if (!arena ||
        !ggml_backend_hrx_ensure_staging_buffer_locked(context->device_context, arena, ggml_backend_hrx_staging_arena_capacity())) {
        hrx_stream_release(stream);
        return false;
    }

    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t uploaded = 0;
    bool ok = true;
    while (uploaded < size) {
        size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
        if (staging_offset >= arena->capacity) {
            ok = GGML_HRX_CHECK(hrx_stream_flush(stream)) && GGML_HRX_CHECK(hrx_stream_wait(stream));
            if (!ok) {
                break;
            }
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
            staging_offset = 0;
        }

        const size_t available = arena->capacity - staging_offset;
        const size_t chunk_size = std::min(size - uploaded, available);
        if (chunk_size == 0) {
            GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
            ok = false;
            break;
        }

        std::memcpy(arena->mapped + staging_offset, bytes + uploaded, chunk_size);
        ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
            stream,
            arena->buffer,
            staging_offset,
            context->buffer,
            buffer_offset + uploaded,
            chunk_size));
        if (!ok) {
            break;
        }

        arena->offset = ggml_backend_hrx_align_up(staging_offset + chunk_size, GGML_HRX_ALIGNMENT);
        uploaded += chunk_size;
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_copy_tensor_to_staging(
        ggml_backend_hrx_buffer_context * context,
        const ggml_tensor * tensor,
        size_t buffer_offset,
        size_t buffer_size,
        void * data,
        size_t size,
        const char * reason) {
    GGML_UNUSED(reason);
    if (!context || !context->buffer || !data) {
        return false;
    }
    if (buffer_offset > buffer_size || size > buffer_size - buffer_offset) {
        GGML_LOG_ERROR(
            "%s: readback for tensor %s exceeds HRX buffer bounds: offset=%zu size=%zu buffer_size=%zu\n",
            __func__, tensor ? tensor->name : "<unknown>", buffer_offset, size, buffer_size);
        return false;
    }

    hrx_stream_t stream = ggml_backend_hrx_retain_timeline_stream(context->device_context);
    if (!stream) {
        GGML_LOG_ERROR("%s: no HRX stream available for tensor readback\n", __func__);
        return false;
    }

    auto * out_bytes = static_cast<uint8_t *>(data);
    size_t copied = 0;
    bool ok = true;
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        auto * arena = ggml_backend_hrx_get_staging_arena_locked(context->device_context, stream);
        if (!arena ||
            !ggml_backend_hrx_ensure_staging_buffer_locked(context->device_context, arena, ggml_backend_hrx_staging_arena_capacity())) {
            hrx_stream_release(stream);
            return false;
        }

        while (copied < size) {
            size_t staging_offset = ggml_backend_hrx_align_up(arena->offset, GGML_HRX_ALIGNMENT);
            if (staging_offset >= arena->capacity) {
                ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
                if (!ok) {
                    break;
                }
                ggml_backend_hrx_reset_staging_arena_locked(*arena);
                staging_offset = 0;
            }

            const size_t chunk_size = std::min(size - copied, arena->capacity - staging_offset);
            if (chunk_size == 0) {
                GGML_LOG_ERROR("%s: HRX staging arena has no available space\n", __func__);
                ok = false;
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_copy_buffer(
                stream,
                context->buffer,
                buffer_offset + copied,
                arena->buffer,
                staging_offset,
                chunk_size));
            if (!ok) {
                break;
            }

            ok = GGML_HRX_CHECK(hrx_stream_synchronize(stream));
            if (!ok) {
                break;
            }
            std::memcpy(out_bytes + copied, arena->mapped + staging_offset, chunk_size);
            copied += chunk_size;
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }

    hrx_stream_release(stream);
    return ok;
}

static bool ggml_backend_hrx_should_validate_dynamic_index() {
    const char * value = std::getenv("GGML_HRX_VALIDATE_DYNAMIC_INDEX");
    return value && value[0] != '\0' && std::strcmp(value, "0") != 0;
}

static size_t ggml_backend_hrx_tensor_span_size(const ggml_tensor * tensor) {
    size_t span = ggml_type_size(tensor->type);
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 0) {
            span += static_cast<size_t>(tensor->ne[i] - 1) * tensor->nb[i];
        }
    }
    return span;
}

static bool ggml_backend_hrx_read_tensor_bytes(
        ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * tensor,
        std::vector<uint8_t> * data,
        const char * reason) {
    GGML_UNUSED(device_context);
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (!buffer || buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        GGML_LOG_ERROR("%s: tensor %s is not backed by a HRX buffer\n", __func__, ggml_get_name(tensor));
        return false;
    }

    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    const size_t offset = ggml_backend_hrx_tensor_offset(context, tensor);
    const size_t size = ggml_backend_hrx_tensor_span_size(tensor);
    if (offset > buffer->size || size > buffer->size - offset) {
        GGML_LOG_ERROR(
            "%s: tensor %s span is out of bounds offset=%zu size=%zu buffer=%zu\n",
            __func__, ggml_get_name(tensor), offset, size, buffer->size);
        return false;
    }

    data->resize(size);
    return ggml_backend_hrx_copy_tensor_to_staging(
        context, tensor, offset, buffer->size, data->data(), size, reason);
}

static bool ggml_backend_hrx_validate_get_rows_indices(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    if (!ggml_backend_hrx_should_validate_dynamic_index()) {
        return true;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * idx = dst->src[1];
    std::vector<uint8_t> bytes;
    if (!ggml_backend_hrx_read_tensor_bytes(
            context->device_context, idx, &bytes, "validate_get_rows_indices")) {
        return false;
    }

    const int64_t row_limit = src0->ne[1];
    int32_t min_value = INT32_MAX;
    int32_t max_value = INT32_MIN;
    int64_t invalid_count = 0;
    int64_t first_invalid_linear = -1;
    int32_t first_invalid_value = 0;
    for (int64_t i3 = 0; i3 < idx->ne[3]; ++i3) {
        for (int64_t i2 = 0; i2 < idx->ne[2]; ++i2) {
            for (int64_t i1 = 0; i1 < idx->ne[1]; ++i1) {
                for (int64_t i0 = 0; i0 < idx->ne[0]; ++i0) {
                    const size_t byte_offset =
                        static_cast<size_t>(i0) * idx->nb[0] +
                        static_cast<size_t>(i1) * idx->nb[1] +
                        static_cast<size_t>(i2) * idx->nb[2] +
                        static_cast<size_t>(i3) * idx->nb[3];
                    int32_t value = 0;
                    std::memcpy(&value, bytes.data() + byte_offset, sizeof(value));
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                    if (value < 0 || value >= row_limit) {
                        if (invalid_count == 0) {
                            first_invalid_linear = ((i3 * idx->ne[2] + i2) * idx->ne[1] + i1) * idx->ne[0] + i0;
                            first_invalid_value = value;
                        }
                        invalid_count++;
                    }
                }
            }
        }
    }

    if (invalid_count != 0) {
        GGML_LOG_ERROR(
            "%s: invalid GET_ROWS index node=%s idx=%s src=%s row_limit=%" PRId64
            " invalid=%" PRId64 " first_linear=%" PRId64 " first_value=%d min=%d max=%d\n",
            __func__, ggml_get_name(dst), ggml_get_name(idx), ggml_get_name(src0),
            row_limit, invalid_count, first_invalid_linear, first_invalid_value, min_value, max_value);
        return false;
    }
    return true;
}

static size_t ggml_backend_hrx_total_memory(hrx_device_t device) {
    uint64_t memory_total = 0;
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_TOTAL_MEMORY,
            &memory_total, sizeof(memory_total)))) {
        return 0;
    }
    return static_cast<size_t>(memory_total);
}

static std::string ggml_backend_hrx_device_architecture(hrx_device_t device) {
    std::array<char, 128> architecture = {};
    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        return std::string();
    }
    return std::string(architecture.data());
}

static std::string ggml_backend_hrx_device_description(hrx_device_t device) {
    std::array<char, 128> name = {};
    std::array<char, 128> architecture = {};

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_NAME, name.data(), name.size()))) {
        std::snprintf(name.data(), name.size(), "unknown");
    }

    if (!GGML_HRX_CHECK(hrx_device_get_property(
            device, HRX_DEVICE_PROPERTY_ARCHITECTURE,
            architecture.data(), architecture.size()))) {
        std::snprintf(architecture.data(), architecture.size(), "unknown");
    }

    std::string description(name.data());
    if (!description.empty() && architecture[0] != '\0') {
        description += " (";
        description += architecture.data();
        description += ")";
    }
    return description.empty() ? std::string("HRX GPU") : description;
}

static const char * ggml_backend_hrx_kernel_gfx_target(const ggml_backend_hrx_device_context * device_context) {
    GGML_UNUSED(device_context);
    return "gfx1100";
}

static bool ggml_backend_hrx_load_catalog_provider(
        ggml_backend_hrx_device_context * device_context,
        const char * name,
        ggml_backend_hrx_op_provider * provider) {
    const ggml_hrx_kernel_entry * entry =
        ggml_hrx_kernel_catalog_find(name, ggml_backend_hrx_kernel_gfx_target(device_context));
    if (!entry || !entry->data || entry->data_size == 0) {
        return false;
    }

    hrx_executable_t executable = nullptr;
    if (!GGML_HRX_CHECK(hrx_executable_load_data(
            device_context->device, entry->data, entry->data_size, entry->format, &executable))) {
        GGML_LOG_WARN("%s: failed to load HRX catalog kernel %s for %s\n",
            __func__, entry->name, ggml_backend_hrx_kernel_gfx_target(device_context));
        return false;
    }

    uint32_t export_ordinal = 0;
    hrx_executable_export_info_t export_info = {};
    const bool ok = GGML_HRX_CHECK(hrx_executable_lookup_export_by_name(
                        executable, entry->name, &export_ordinal)) &&
                    GGML_HRX_CHECK(hrx_executable_export_info(
                        executable, export_ordinal, &export_info)) &&
                    export_info.binding_count == entry->binding_count &&
                    export_info.parameter_count == entry->parameter_count &&
                    export_info.constant_count * sizeof(uint32_t) == entry->constants_size;
    if (!ok) {
        GGML_LOG_WARN(
            "%s: HRX catalog kernel %s has unsupported ABI "
            "(bindings=%u expected=%u constants=%u constants_size=%u parameters=%u expected_parameters=%u workgroup=%ux%ux%u)\n",
            __func__,
            entry->name,
            export_info.binding_count,
            entry->binding_count,
            export_info.constant_count,
            entry->constants_size,
            export_info.parameter_count,
            entry->parameter_count,
            export_info.workgroup_size[0],
            export_info.workgroup_size[1],
            export_info.workgroup_size[2]);
        hrx_executable_release(executable);
        return false;
    }

    provider->kind = ggml_backend_hrx_provider_kind::hsaco;
    provider->executable = executable;
    provider->export_ordinal = export_ordinal;
    provider->export_info = export_info;
    provider->export_info.workgroup_size[0] = entry->workgroup_size[0];
    provider->export_info.workgroup_size[1] = entry->workgroup_size[1];
    provider->export_info.workgroup_size[2] = entry->workgroup_size[2];
    return true;
}

static bool ggml_backend_hrx_load_rms_norm_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_rms_norm_f32", &device_context->rms_norm_provider);
}

static bool ggml_backend_hrx_load_rms_norm_mul_providers(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_rms_norm_mul_f32", &device_context->rms_norm_mul_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_rms_norm_mul_wg128_f32", &device_context->rms_norm_mul_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_rms_norm_mul_rope_f32", &device_context->rms_norm_mul_rope_f32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context,
        "hrx_rms_norm_mul_rope_set_rows_f32_f16",
        &device_context->rms_norm_mul_rope_set_rows_f32_f16_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_add_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_add_f32", &device_context->add_provider);
}

static bool ggml_backend_hrx_load_mul_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_mul_f32", &device_context->mul_provider);
}

static bool ggml_backend_hrx_load_add_broadcast_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_add_f32_broadcast", &device_context->add_broadcast_provider);
}

static bool ggml_backend_hrx_load_mul_broadcast_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_mul_f32_broadcast", &device_context->mul_broadcast_provider);
}

static bool ggml_backend_hrx_load_div_broadcast_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_div_f32_broadcast", &device_context->div_broadcast_provider);
}

static bool ggml_backend_hrx_load_add8_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_add8_f32", &device_context->add8_provider);
}

static bool ggml_backend_hrx_load_mul_sum8_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_mul_sum8_f32", &device_context->mul_sum8_provider);
}

static bool ggml_backend_hrx_load_mul_add_add_broadcast_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_add_add_f32_broadcast", &device_context->mul_add_add_broadcast_provider);
}

static bool ggml_backend_hrx_load_scale_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_scale_f32", &device_context->scale_provider);
}

static bool ggml_backend_hrx_load_set_rows_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_set_rows_f32_f32", &device_context->set_rows_f32_provider);
}

static bool ggml_backend_hrx_load_set_rows_f16_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_set_rows_f32_f16", &device_context->set_rows_f16_provider);
}

static bool ggml_backend_hrx_load_set_rows_q8_0_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_set_rows_f32_q8_0", &device_context->set_rows_q8_0_provider);
}

static bool ggml_backend_hrx_load_set_rows_q4_0_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_set_rows_f32_q4_0", &device_context->set_rows_q4_0_provider);
}

static bool ggml_backend_hrx_load_silu_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_silu_f32", &device_context->silu_provider);
}

static bool ggml_backend_hrx_load_sigmoid_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_sigmoid_f32", &device_context->sigmoid_provider);
}

static bool ggml_backend_hrx_load_softplus_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_softplus_f32", &device_context->softplus_provider);
}

static bool ggml_backend_hrx_load_swiglu_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_swiglu_f32", &device_context->swiglu_provider);
}

static bool ggml_backend_hrx_load_sum_rows_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_sum_rows_f32", &device_context->sum_rows_provider);
}

static bool ggml_backend_hrx_load_l2_norm_provider(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_l2_norm_f32", &device_context->l2_norm_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_l2_norm_wg128_f32", &device_context->l2_norm_wg128_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_clamp_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_clamp_f32", &device_context->clamp_provider);
}

static bool ggml_backend_hrx_load_get_rows_f32_provider(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_get_rows_f32", &device_context->get_rows_f32_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_get_rows_f32_nr1", &device_context->get_rows_f32_nr1_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_get_rows_q5_k_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_get_rows_q5_k_f32", &device_context->get_rows_q5_k_provider);
}

static bool ggml_backend_hrx_load_mul_mat_vec_providers(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_f32", &device_context->mul_mat_vec_bf16_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_wg128_f32", &device_context->mul_mat_vec_bf16_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_wg64_f32", &device_context->mul_mat_vec_bf16_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_cols1_f32", &device_context->mul_mat_vec_bf16_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_rows2_cols1_f32",
        &device_context->mul_mat_vec_bf16_rows2_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_rows2_cols1_wg32_f32",
        &device_context->mul_mat_vec_bf16_rows2_cols1_wg32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32",
        &device_context->mul_mat_vec_bf16_rows4_k512_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32",
        &device_context->mul_mat_vec_bf16_rows4_k2048_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_cols4_f32", &device_context->mul_mat_vec_bf16_cols4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_cols8_f32", &device_context->mul_mat_vec_bf16_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_cols16_f32", &device_context->mul_mat_vec_bf16_cols16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_cols32_f32", &device_context->mul_mat_vec_bf16_cols32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_rows2_cols16_f32",
        &device_context->mul_mat_vec_bf16_rows2_cols16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_wmma16x16_f32",
        &device_context->mul_mat_vec_bf16_wmma16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_f32",
        &device_context->mul_mat_vec_bf16_swiglu_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_wg128_f32",
        &device_context->mul_mat_vec_bf16_swiglu_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_wg64_f32",
        &device_context->mul_mat_vec_bf16_swiglu_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_cols1_f32",
        &device_context->mul_mat_vec_bf16_swiglu_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_rows2_cols1_f32",
        &device_context->mul_mat_vec_bf16_swiglu_rows2_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32",
        &device_context->mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_cols4_f32",
        &device_context->mul_mat_vec_bf16_swiglu_cols4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_cols8_f32",
        &device_context->mul_mat_vec_bf16_swiglu_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_cols16_f32",
        &device_context->mul_mat_vec_bf16_swiglu_cols16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_rows2_cols8_f32",
        &device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_swiglu_wmma16x16_f32",
        &device_context->mul_mat_vec_bf16_swiglu_wmma16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_bf16_set_rows_f16",
        &device_context->mul_mat_vec_bf16_set_rows_f16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_f32", &device_context->mul_mat_vec_f16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_batched_f32", &device_context->mul_mat_vec_f16_batched_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_batched_cols1_f32",
        &device_context->mul_mat_vec_f16_batched_cols1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_batched_cols4_f32",
        &device_context->mul_mat_vec_f16_batched_cols4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_batched_cols8_f32",
        &device_context->mul_mat_vec_f16_batched_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f16_batched_cols16_f32",
        &device_context->mul_mat_vec_f16_batched_cols16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_f32", &device_context->mul_mat_vec_f32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_f32", &device_context->mul_mat_vec_f32_batched_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_f32",
        &device_context->mul_mat_vec_f32_batched_cols1_ne2_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32",
        &device_context->mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_cols8_f32",
        &device_context->mul_mat_vec_f32_batched_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_cols16_f32",
        &device_context->mul_mat_vec_f32_batched_cols16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_f32_batched_rows2_cols8_f32",
        &device_context->mul_mat_vec_f32_batched_rows2_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_quantize_q8_1_f32", &device_context->quantize_q8_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_quantize_q8_1_x4_f32", &device_context->quantize_q8_1_x4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q4_k_f32", &device_context->mul_mat_vec_q4_k_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q4_k_q8_1_f32", &device_context->mul_mat_vec_q4_k_q8_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_f32", &device_context->mul_mat_vec_q5_k_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_q8_1_f32", &device_context->mul_mat_vec_q5_k_q8_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32",
        &device_context->mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32",
        &device_context->mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32",
        &device_context->mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32",
        &device_context->mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q6_k_f32", &device_context->mul_mat_vec_q6_k_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q6_k_q8_1_f32", &device_context->mul_mat_vec_q6_k_q8_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32",
        &device_context->mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32",
        &device_context->mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_f32", &device_context->mul_mat_vec_q8_0_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_cols8_f32",
        &device_context->mul_mat_vec_q8_0_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32",
        &device_context->mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_add_f32",
        &device_context->mul_mat_vec_q8_0_add_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_add_cols8_f32",
        &device_context->mul_mat_vec_q8_0_add_cols8_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32",
        &device_context->mul_mat_vec_q8_0_add_rows4_cols4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32",
        &device_context->mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_mul_mat_id_providers(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_f32", &device_context->mul_mat_id_q4_k_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_wg64_f32", &device_context->mul_mat_id_q4_k_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_row4_wg64_f32",
        &device_context->mul_mat_id_q4_k_row4_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_clear_u32", &device_context->clear_u32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_compact_moe_routes_i32", &device_context->compact_moe_routes_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32",
        &device_context->mul_mat_id_q4_k_grouped_row2_route8_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32",
        &device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_q8_1_f32",
        &device_context->mul_mat_id_q4_k_q8_1_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_mul_f32",
        &device_context->mul_mat_id_q4_k_mul_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_mul_wg64_f32",
        &device_context->mul_mat_id_q4_k_mul_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_mul_packed_wg64_f32",
        &device_context->mul_mat_id_q4_k_mul_packed_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32",
        &device_context->mul_mat_id_q4_k_mul_rows2_x16_wg32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_f32",
        &device_context->mul_mat_id_q4_k_swiglu_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_wg64_f32",
        &device_context->mul_mat_id_q4_k_swiglu_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32",
        &device_context->mul_mat_id_q4_k_swiglu_row4_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32",
        &device_context->mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32",
        &device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32",
        &device_context->mul_mat_id_q4_k_swiglu_packed_wg64_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_mul_mat_id_q4_k_mul_q8_1_f32",
        &device_context->mul_mat_id_q4_k_mul_q8_1_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_flash_attn_ext_providers(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_f16_decode", &device_context->flash_attn_ext_f16_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_f16_prefill_tile8",
        &device_context->flash_attn_ext_f16_prefill_tile_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_f16_prefill_wmma16",
        &device_context->flash_attn_ext_f16_prefill_wmma_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_f16_prefill_direct",
        &device_context->flash_attn_ext_f16_prefill_direct_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_bf16_decode", &device_context->flash_attn_ext_bf16_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_f32_decode", &device_context->flash_attn_ext_f32_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_q4_0_decode", &device_context->flash_attn_ext_q4_0_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_q8_0_decode", &device_context->flash_attn_ext_q8_0_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_flash_attn_ext_f32_q8_0_q4_0_decode",
        &device_context->flash_attn_ext_q8_0_q4_0_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_concat_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_concat_f32", &device_context->concat_f32_provider);
}

static bool ggml_backend_hrx_load_copy_strided_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_copy_strided_f32", &device_context->copy_strided_f32_provider);
}

static bool ggml_backend_hrx_load_copy_f32_f16_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_copy_f32_f16", &device_context->copy_f32_f16_provider);
}

static bool ggml_backend_hrx_load_soft_max_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_soft_max_f32", &device_context->soft_max_f32_provider);
}

static bool ggml_backend_hrx_load_soft_max_f32_mask_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_soft_max_f32_mask", &device_context->soft_max_f32_mask_provider);
}

static bool ggml_backend_hrx_load_argsort_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_argsort_f32_i32", &device_context->argsort_f32_provider);
}

static bool ggml_backend_hrx_load_topk_moe_f32_providers(ggml_backend_hrx_device_context * device_context) {
    bool ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_topk_moe_f32", &device_context->topk_moe_f32_provider);
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_topk_moe_f32_shared4", &device_context->topk_moe_f32_shared4_provider) || ok;
    ok = ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_topk_moe_f32_wave32", &device_context->topk_moe_f32_wave32_provider) || ok;
    return ok;
}

static bool ggml_backend_hrx_load_rope_f32_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_rope_f32", &device_context->rope_f32_provider);
}

static bool ggml_backend_hrx_load_rope_set_rows_f32_f16_provider(
        ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_rope_set_rows_f32_f16", &device_context->rope_set_rows_f32_f16_provider);
}

static bool ggml_backend_hrx_load_ssm_conv_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(device_context, "hrx_ssm_conv_f32", &device_context->ssm_conv_provider);
}

static bool ggml_backend_hrx_load_ssm_conv_update_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_ssm_conv_update_f32", &device_context->ssm_conv_update_provider);
}

static bool ggml_backend_hrx_load_gated_delta_net_provider(ggml_backend_hrx_device_context * device_context) {
    return ggml_backend_hrx_load_catalog_provider(
        device_context, "hrx_gated_delta_net_f32", &device_context->gated_delta_net_provider);
}

static const char * ggml_backend_hrx_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return ggml_backend_hrx_get_buft_context(buft)->name.c_str();
}

static void ggml_backend_hrx_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (context->buffer) {
        hrx_buffer_release(context->buffer);
    }
    delete context;
}

static void * ggml_backend_hrx_buffer_get_base(ggml_backend_buffer_t buffer) {
    return ggml_backend_hrx_get_buffer_context(buffer)->base;
}

static void ggml_backend_hrx_buffer_memset_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, buffer_offset, size, &value, sizeof(value));
}

static void ggml_backend_hrx_buffer_set_tensor(
        ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_stage_and_copy_tensor(context, tensor, data, buffer_offset, buffer->size, size)) {
        GGML_LOG_ERROR("%s: failed to upload tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static void ggml_backend_hrx_buffer_get_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (size == 0 || !context->buffer) {
        return;
    }

    const size_t buffer_offset = ggml_backend_hrx_tensor_offset(context, tensor) + offset;
    if (!ggml_backend_hrx_copy_tensor_to_staging(
            context, tensor, buffer_offset, buffer->size, data, size, "get_tensor")) {
        GGML_LOG_ERROR("%s: failed to read tensor %s through HRX staging\n", __func__, tensor->name);
    }
}

static bool ggml_backend_hrx_buffer_cpy_tensor(
        ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t src_buffer = src->view_src ? src->view_src->buffer : src->buffer;
    if (!src_buffer || src_buffer->iface.get_base != ggml_backend_hrx_buffer_get_base) {
        return false;
    }

    auto * dst_context = ggml_backend_hrx_get_buffer_context(buffer);
    auto * src_context = ggml_backend_hrx_get_buffer_context(src_buffer);
    if (dst_context->device_context != src_context->device_context ||
        !dst_context->buffer || !src_context->buffer) {
        return false;
    }

    if (!ggml_backend_hrx_sync_streams(dst_context->device_context)) {
        return false;
    }

    const size_t src_offset = ggml_backend_hrx_tensor_offset(src_context, src);
    const size_t dst_offset = ggml_backend_hrx_tensor_offset(dst_context, dst);
    const size_t size = ggml_nbytes(src);
    return ggml_backend_hrx_queue_copy_stream_sync(
        dst_context->device_context,
        src_context->buffer, src_offset,
        dst_context->buffer, dst_offset,
        size);
}

static void ggml_backend_hrx_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    auto * context = ggml_backend_hrx_get_buffer_context(buffer);
    if (buffer->size == 0 || !context->buffer) {
        return;
    }

    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return;
    }

    (void) ggml_backend_hrx_queue_fill_stream_sync(
        context->device_context, context->buffer, 0, buffer->size, &value, sizeof(value));
}

static const ggml_backend_buffer_i ggml_backend_hrx_buffer_i = {
    /* .free_buffer   = */ ggml_backend_hrx_buffer_free_buffer,
    /* .get_base      = */ ggml_backend_hrx_buffer_get_base,
    /* .init_tensor   = */ nullptr,
    /* .memset_tensor = */ ggml_backend_hrx_buffer_memset_tensor,
    /* .set_tensor    = */ ggml_backend_hrx_buffer_set_tensor,
    /* .get_tensor    = */ ggml_backend_hrx_buffer_get_tensor,
    /* .cpy_tensor    = */ ggml_backend_hrx_buffer_cpy_tensor,
    /* .clear         = */ ggml_backend_hrx_buffer_clear,
    /* .reset         = */ nullptr,
};

static ggml_backend_buffer_t ggml_backend_hrx_buffer_type_alloc_buffer(
        ggml_backend_buffer_type_t buft, size_t size) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);

    hrx_buffer_t hrx_buffer = nullptr;
    if (size > 0 &&
        !GGML_HRX_CHECK(hrx_allocator_allocate_buffer(
            hrx_device_allocator(buft_context->device_context->device),
            buft_context->params, size, &hrx_buffer))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_buffer_context {
        /* .device_context = */ buft_context->device_context,
        /* .buffer         = */ hrx_buffer,
        /* .base           = */ reinterpret_cast<uint8_t *>(GGML_HRX_FAKE_PTR_BASE),
    };
    if (!context) {
        if (hrx_buffer) {
            hrx_buffer_release(hrx_buffer);
        }
        return nullptr;
    }

    ggml_backend_buffer_t buffer = ggml_backend_buffer_init(
        buft, ggml_backend_hrx_buffer_i, context, size);
    if (!buffer) {
        if (context->buffer) {
            hrx_buffer_release(context->buffer);
        }
        delete context;
    }
    return buffer;
}

static size_t ggml_backend_hrx_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return GGML_HRX_ALIGNMENT;
}

static size_t ggml_backend_hrx_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    auto * buft_context = ggml_backend_hrx_get_buft_context(buft);
    return buft_context->device_context->memory_total > 0 ?
        buft_context->device_context->memory_total :
        std::numeric_limits<size_t>::max();
}

static const ggml_backend_buffer_type_i ggml_backend_hrx_buffer_type_i = {
    /* .get_name       = */ ggml_backend_hrx_buffer_type_get_name,
    /* .alloc_buffer   = */ ggml_backend_hrx_buffer_type_alloc_buffer,
    /* .get_alignment  = */ ggml_backend_hrx_buffer_type_get_alignment,
    /* .get_max_size   = */ ggml_backend_hrx_buffer_type_get_max_size,
    /* .get_alloc_size = */ nullptr,
    /* .is_host        = */ nullptr,
};

static ggml_backend_buffer_type_t ggml_backend_hrx_device_buffer_type(ggml_backend_dev_t dev) {
    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    static std::vector<std::unique_ptr<ggml_backend_buffer_type>> buffer_types;
    static std::vector<std::unique_ptr<ggml_backend_hrx_buffer_type_context>> contexts;

    for (const auto & buft : buffer_types) {
        auto * context = ggml_backend_hrx_get_buft_context(buft.get());
        if (context->device_context == device_context) {
            return buft.get();
        }
    }

    auto * context = new ggml_backend_hrx_buffer_type_context {
        /* .device_context = */ device_context,
        /* .name           = */ device_context->name,
        /* .params         = */ {
            /* .type = */ HRX_MEMORY_TYPE_DEVICE_LOCAL,
            /* .access = */ HRX_MEMORY_ACCESS_ALL,
            /* .usage = */ HRX_BUFFER_USAGE_DEFAULT,
            /* .queue_affinity = */ 0,
        },
    };

    auto * buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_hrx_buffer_type_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };

    contexts.emplace_back(context);
    buffer_types.emplace_back(buft);
    return buft;
}

static const char * ggml_backend_hrx_get_name(ggml_backend_t backend) {
    return static_cast<ggml_backend_hrx_context *>(backend->context)->name.c_str();
}

static void ggml_backend_hrx_free(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx_release_scratch_buffers(context);
        ggml_backend_hrx_unregister_stream(context->device_context, context->stream);
        hrx_stream_release(context->stream);
    }
    delete context;
    delete backend;
}

static void ggml_backend_hrx_synchronize(ggml_backend_t backend) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (context->stream) {
        GGML_HRX_CHECK(hrx_stream_synchronize(context->stream));
        ggml_backend_hrx_recycle_scratch_buffers(context);
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        if (auto * arena = ggml_backend_hrx_find_staging_arena_locked(context->device_context, context->stream)) {
            ggml_backend_hrx_reset_staging_arena_locked(*arena);
        }
    }
}

static bool ggml_backend_hrx_approximate_kernels_disabled() {
    return ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FAST_APPROX_PROMPT");
}

static bool ggml_backend_hrx_flash_attn_ext_decode_disabled() {
    return ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FLASH_ATTN_EXT_DECODE");
}

static bool ggml_backend_hrx_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b);

static bool ggml_backend_hrx_supports_rms_norm(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           device_context->rms_norm_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float) &&
           ggml_are_same_shape(src0, op);
}

static bool ggml_backend_hrx_supports_rms_norm_mul(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    if (device_context->rms_norm_mul_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !ggml_backend_hrx_supports_rms_norm(device_context, rms_norm) ||
        !mul ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        !mul->src[0] ||
        !mul->src[1] ||
        !ggml_are_same_shape(rms_norm, mul) ||
        (mul->src[0] != rms_norm && mul->src[1] != rms_norm) ||
        mul->nb[0] != sizeof(float)) {
        return false;
    }

    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    return weight &&
           weight->type == GGML_TYPE_F32 &&
           (weight->ne[0] == rms_norm->ne[0] || weight->ne[0] == 1) &&
           (weight->ne[1] == rms_norm->ne[1] || weight->ne[1] == 1) &&
           (weight->ne[2] == rms_norm->ne[2] || weight->ne[2] == 1) &&
           (weight->ne[3] == rms_norm->ne[3] || weight->ne[3] == 1) &&
           (weight->ne[0] == 1 || weight->nb[0] == sizeof(float));
}

static bool ggml_backend_hrx_supports_binary_elementwise(
        const ggml_backend_hrx_op_provider & provider,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_broadcast_elementwise(
        const ggml_backend_hrx_op_provider & provider,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == op->ne[0] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == op->ne[3] &&
           (src1->ne[0] == src0->ne[0] || src1->ne[0] == 1) &&
           (src1->ne[1] == src0->ne[1] || src1->ne[1] == 1) &&
           (src1->ne[2] == src0->ne[2] || src1->ne[2] == 1) &&
           (src1->ne[3] == src0->ne[3] || src1->ne[3] == 1) &&
           (src0->ne[0] == 1 || src0->nb[0] == sizeof(float)) &&
           (src1->ne[0] == 1 || src1->nb[0] == sizeof(float)) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_supports_add(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ADD")) {
        return false;
    }
    return ggml_backend_hrx_supports_binary_elementwise(device_context->add_provider, op) ||
           ggml_backend_hrx_supports_broadcast_elementwise(device_context->add_broadcast_provider, op);
}

static bool ggml_backend_hrx_supports_mul(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL")) {
        return false;
    }
    return ggml_backend_hrx_supports_binary_elementwise(device_context->mul_provider, op) ||
           ggml_backend_hrx_supports_broadcast_elementwise(device_context->mul_broadcast_provider, op);
}

static bool ggml_backend_hrx_supports_div(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx_supports_broadcast_elementwise(device_context->div_broadcast_provider, op);
}

static bool ggml_backend_hrx_supports_broadcast_operand_f32(
        const ggml_tensor * src,
        const ggml_tensor * shape) {
    return src &&
           shape &&
           src->type == GGML_TYPE_F32 &&
           (src->ne[0] == shape->ne[0] || src->ne[0] == 1) &&
           (src->ne[1] == shape->ne[1] || src->ne[1] == 1) &&
           (src->ne[2] == shape->ne[2] || src->ne[2] == 1) &&
           (src->ne[3] == shape->ne[3] || src->ne[3] == 1) &&
           (src->ne[0] == 1 || src->nb[0] == sizeof(float));
}

static bool ggml_backend_hrx_supports_mul_add_add_broadcast(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mul,
        const ggml_tensor * first_add,
        const ggml_tensor * second_add,
        const ggml_tensor ** add_src0,
        const ggml_tensor ** add_src1) {
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_ADD_ADD_FUSION") ||
        device_context->mul_add_add_broadcast_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !mul ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        !mul->src[0] ||
        !ggml_backend_hrx_supports_broadcast_operand_f32(mul->src[0], mul) ||
        !ggml_backend_hrx_supports_broadcast_operand_f32(mul->src[1], mul) ||
        !ggml_are_same_shape(mul, mul->src[0]) ||
        mul->nb[0] != sizeof(float) ||
        !first_add ||
        first_add->op != GGML_OP_ADD ||
        first_add->type != GGML_TYPE_F32 ||
        (first_add->src[0] != mul && first_add->src[1] != mul) ||
        !ggml_are_same_shape(mul, first_add) ||
        first_add->nb[0] != sizeof(float) ||
        !second_add ||
        second_add->op != GGML_OP_ADD ||
        second_add->type != GGML_TYPE_F32 ||
        (second_add->src[0] != first_add && second_add->src[1] != first_add) ||
        !ggml_are_same_shape(mul, second_add) ||
        second_add->nb[0] != sizeof(float)) {
        return false;
    }

    const ggml_tensor * other0 = first_add->src[0] == mul ? first_add->src[1] : first_add->src[0];
    const ggml_tensor * other1 = second_add->src[0] == first_add ? second_add->src[1] : second_add->src[0];
    if (!ggml_backend_hrx_supports_broadcast_operand_f32(other0, mul) ||
        !ggml_backend_hrx_supports_broadcast_operand_f32(other1, mul)) {
        return false;
    }

    *add_src0 = other0;
    *add_src1 = other1;
    return true;
}

static bool ggml_backend_hrx_supports_add8_tensor(
        const ggml_tensor * tensor,
        const ggml_tensor * shape) {
    return tensor &&
           tensor->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(tensor, shape) &&
           tensor->ne[2] == 1 &&
           tensor->ne[3] == 1 &&
           tensor->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_try_collect_add8_chain(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int start,
        std::array<const ggml_tensor *, 8> * sources,
        const ggml_tensor ** dst) {
    static constexpr int ADD_COUNT = 7;
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        device_context->add8_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        start + ADD_COUNT > cgraph->n_nodes) {
        return false;
    }

    const ggml_tensor * current = cgraph->nodes[start];
    if (!current ||
        current->op != GGML_OP_ADD ||
        !ggml_backend_hrx_supports_add8_tensor(current, current)) {
        return false;
    }

    std::array<int, ADD_COUNT> idxs = {};
    std::array<ggml_op, ADD_COUNT> ops = {};
    idxs[0] = start;
    ops[0] = GGML_OP_ADD;
    (*sources)[0] = current->src[0];
    (*sources)[1] = current->src[1];

    for (int add_idx = 1; add_idx < ADD_COUNT; ++add_idx) {
        const ggml_tensor * next = cgraph->nodes[start + add_idx];
        if (!next ||
            next->op != GGML_OP_ADD ||
            next->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(next, current) ||
            !ggml_is_contiguous(next)) {
            return false;
        }

        const ggml_tensor * term = nullptr;
        if (next->src[0] == current) {
            term = next->src[1];
        } else if (next->src[1] == current) {
            term = next->src[0];
        } else {
            return false;
        }

        (*sources)[add_idx + 1] = term;
        current = next;
        idxs[add_idx] = start + add_idx;
        ops[add_idx] = GGML_OP_ADD;
    }

    for (const ggml_tensor * src : *sources) {
        if (!ggml_backend_hrx_supports_add8_tensor(src, current)) {
            return false;
        }
    }

    const int outputs[1] = { start + ADD_COUNT - 1 };
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs.data(), ADD_COUNT, ops.data(), outputs, 1)) {
        return false;
    }

    *dst = current;
    return true;
}

static bool ggml_backend_hrx_is_reshape_view(const ggml_tensor * op) {
    return op && (op->op == GGML_OP_RESHAPE || op->op == GGML_OP_VIEW);
}

static const ggml_tensor * ggml_backend_hrx_unwrap_reshape_view_src0(const ggml_tensor * op) {
    while (ggml_backend_hrx_is_reshape_view(op)) {
        op = op->src[0];
    }
    return op;
}

static bool ggml_backend_hrx_supports_mul_sum8(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mul,
        const std::array<const ggml_tensor *, 8> & sources,
        const ggml_tensor * dst) {
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_SUM8_FUSION") ||
        device_context->mul_sum8_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !mul ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        !mul->src[0] ||
        !mul->src[1] ||
        mul->src[0]->type != GGML_TYPE_F32 ||
        mul->src[1]->type != GGML_TYPE_F32 ||
        mul->ne[1] != 8 ||
        mul->ne[3] != 1 ||
        mul->nb[0] != sizeof(float) ||
        mul->src[0]->ne[0] != mul->ne[0] ||
        mul->src[0]->ne[1] != mul->ne[1] ||
        mul->src[0]->ne[2] != mul->ne[2] ||
        mul->src[0]->ne[3] != mul->ne[3] ||
        mul->src[0]->nb[0] != sizeof(float) ||
        mul->src[1]->ne[0] != 1 ||
        mul->src[1]->ne[1] != mul->ne[1] ||
        mul->src[1]->ne[2] != mul->ne[2] ||
        mul->src[1]->ne[3] != 1 ||
        mul->src[1]->nb[0] != sizeof(float) ||
        !dst ||
        dst->type != GGML_TYPE_F32 ||
        dst->ne[0] != mul->ne[0] ||
        dst->ne[1] != mul->ne[2] ||
        dst->ne[2] != 1 ||
        dst->ne[3] != 1 ||
        dst->nb[0] != sizeof(float)) {
        return false;
    }

    std::array<bool, 8> seen_expert = {};
    for (const ggml_tensor * src : sources) {
        if (!src || src->type != GGML_TYPE_F32 ||
            src->ne[0] != dst->ne[0] ||
            src->ne[1] != dst->ne[1] ||
            src->ne[2] != 1 ||
            src->ne[3] != 1 ||
            src->nb[0] != sizeof(float) ||
            src->nb[1] != mul->nb[2] ||
            ggml_backend_hrx_unwrap_reshape_view_src0(src) != mul ||
            src->view_src != mul ||
            src->view_offs % static_cast<size_t>(mul->nb[1]) != 0) {
            return false;
        }

        const size_t expert = src->view_offs / static_cast<size_t>(mul->nb[1]);
        if (expert >= seen_expert.size() ||
            src->view_offs != expert * static_cast<size_t>(mul->nb[1]) ||
            seen_expert[expert]) {
            return false;
        }
        seen_expert[expert] = true;
    }
    return true;
}

static bool ggml_backend_hrx_find_mul_sum8_fusion(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_cgraph * cgraph,
        int mul_idx,
        std::array<const ggml_tensor *, 8> * sources,
        const ggml_tensor ** dst,
        int * last_idx) {
    const ggml_tensor * mul = cgraph->nodes[mul_idx];
    for (int i = mul_idx + 1; i < cgraph->n_nodes && i < mul_idx + 16; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        if (ggml_backend_hrx_is_reshape_view(node)) {
            continue;
        }
        if (!node || node->op != GGML_OP_ADD) {
            return false;
        }
        std::array<const ggml_tensor *, 8> candidate_sources = {};
        const ggml_tensor * candidate_dst = nullptr;
        if (ggml_backend_hrx_try_collect_add8_chain(
                device_context, cgraph, i, &candidate_sources, &candidate_dst) &&
            ggml_backend_hrx_supports_mul_sum8(device_context, mul, candidate_sources, candidate_dst)) {
            std::vector<int> idxs;
            std::vector<ggml_op> ops;
            idxs.reserve(static_cast<size_t>(i + 7 - mul_idx));
            ops.reserve(static_cast<size_t>(i + 7 - mul_idx));
            for (int idx = mul_idx; idx < i + 7; ++idx) {
                idxs.push_back(idx);
                ops.push_back(cgraph->nodes[idx]->op);
            }
            const int outputs[1] = { i + 6 };
            if (!ggml_can_fuse_subgraph_ext(
                    cgraph,
                    idxs.data(),
                    static_cast<int>(idxs.size()),
                    ops.data(),
                    outputs,
                    1)) {
                return false;
            }
            *sources = candidate_sources;
            *dst = candidate_dst;
            *last_idx = i + 6;
            return true;
        }
        return false;
    }
    return false;
}

static bool ggml_backend_hrx_supports_scale(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->scale_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_unary_provider(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (op->op != GGML_OP_UNARY || ggml_backend_hrx_approximate_kernels_disabled()) {
        return nullptr;
    }
    switch (ggml_get_unary_op(op)) {
        case GGML_UNARY_OP_SILU:
            return &device_context->silu_provider;
        case GGML_UNARY_OP_SIGMOID:
            return &device_context->sigmoid_provider;
        case GGML_UNARY_OP_SOFTPLUS:
            return &device_context->softplus_provider;
        default:
            return nullptr;
    }
}

static bool ggml_backend_hrx_supports_unary_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_unary_provider(device_context, op);
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_swiglu_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_SWIGLU") &&
           device_context->swiglu_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           op->op == GGML_OP_GLU &&
           ggml_get_glu_op(op) == GGML_GLU_OP_SWIGLU &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, src1) &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_sum_rows(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           device_context->sum_rows_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == 1 &&
           op->ne[1] == src0->ne[1] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           src0->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_supports_l2_norm(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        !src0 ||
        src0->type != GGML_TYPE_F32 ||
        op->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(src0, op) ||
        src0->nb[0] != sizeof(float) ||
        op->nb[0] != sizeof(float)) {
        return false;
    }

    if (src0->ne[0] <= 128 &&
        device_context->l2_norm_wg128_provider.kind == ggml_backend_hrx_provider_kind::hsaco) {
        return true;
    }
    return device_context->l2_norm_provider.kind == ggml_backend_hrx_provider_kind::hsaco;
}

static bool ggml_backend_hrx_supports_clamp(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return device_context->clamp_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_cpy(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1 ||
        ggml_nelements(src0) != ggml_nelements(op) ||
        ggml_nbytes(src1) != ggml_nbytes(op) ||
        !ggml_is_contiguous(op)) {
        return false;
    }

    if (src0->type == GGML_TYPE_F32 &&
        src1->type == GGML_TYPE_F16 &&
        op->type == GGML_TYPE_F16) {
        return ggml_is_contiguous(src0) &&
               device_context->copy_f32_f16_provider.kind == ggml_backend_hrx_provider_kind::hsaco;
    }

    return src0->type == src1->type &&
           src0->type == op->type &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           (ggml_is_contiguous(src0) || src0->nb[0] == ggml_type_size(src0->type));
}

static bool ggml_backend_hrx_supports_cont(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    return src0 &&
           src0->type == op->type &&
           ggml_nelements(src0) == ggml_nelements(op) &&
           ggml_row_size(src0->type, src0->ne[0]) * ggml_nrows(src0) == ggml_nbytes(op) &&
           ggml_is_contiguous(op) &&
           (ggml_is_contiguous(src0) || src0->nb[0] == ggml_type_size(src0->type));
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_set_rows_provider(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    switch (op->type) {
        case GGML_TYPE_F32:
            return &device_context->set_rows_f32_provider;
        case GGML_TYPE_F16:
            return &device_context->set_rows_f16_provider;
        case GGML_TYPE_Q8_0:
            return &device_context->set_rows_q8_0_provider;
        case GGML_TYPE_Q4_0:
            return &device_context->set_rows_q4_0_provider;
        default:
            return nullptr;
    }
}

static bool ggml_backend_hrx_supports_set_rows(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    const ggml_backend_hrx_op_provider * provider =
        ggml_backend_hrx_set_rows_provider(device_context, op);
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 && src1 && src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I64 &&
           src2->type == op->type &&
           op->ne[0] == src0->ne[0] &&
           op->ne[2] == src0->ne[2] &&
           op->ne[3] == src0->ne[3] &&
           src0->ne[1] == src1->ne[0] &&
           src0->ne[2] % src1->ne[1] == 0 &&
           src0->ne[3] % src1->ne[2] == 0 &&
           src1->ne[3] == 1 &&
           (!ggml_is_quantized(op->type) || src0->ne[0] % ggml_blck_size(op->type) == 0) &&
           ggml_is_contiguous_rows(src0) &&
           ggml_is_contiguous_rows(op);
}

static bool ggml_backend_hrx_supports_get_rows_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const bool has_provider =
        (src0 && src0->type == GGML_TYPE_F32 &&
         device_context->get_rows_f32_provider.kind == ggml_backend_hrx_provider_kind::hsaco) ||
        (src0 && src0->type == GGML_TYPE_Q5_K &&
         device_context->get_rows_q5_k_provider.kind == ggml_backend_hrx_provider_kind::hsaco);
    return has_provider &&
           src0 && src1 &&
           (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_Q5_K) &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           op->ne[0] == src0->ne[0] &&
           op->ne[2] == src1->ne[1] &&
           op->ne[3] == src1->ne[2] &&
           src0->ne[2] == src1->ne[1] &&
           src0->ne[3] == src1->ne[2] &&
           (src0->type == GGML_TYPE_Q5_K || src0->nb[0] == sizeof(float)) &&
           src1->nb[0] == sizeof(int32_t) &&
           op->nb[0] == sizeof(float);
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_mul_mat_vec_provider(
        const ggml_backend_hrx_device_context * device_context,
        ggml_type type) {
    switch (type) {
        case GGML_TYPE_BF16:
            return &device_context->mul_mat_vec_bf16_provider;
        case GGML_TYPE_F16:
            return &device_context->mul_mat_vec_f16_provider;
        case GGML_TYPE_F32:
            return &device_context->mul_mat_vec_f32_provider;
        case GGML_TYPE_Q4_K:
            return &device_context->mul_mat_vec_q4_k_provider;
        case GGML_TYPE_Q5_K:
            return &device_context->mul_mat_vec_q5_k_provider;
        case GGML_TYPE_Q6_K:
            return &device_context->mul_mat_vec_q6_k_provider;
        case GGML_TYPE_Q8_0:
            return &device_context->mul_mat_vec_q8_0_provider;
        default:
            return nullptr;
    }
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_mul_mat_vec_batched_provider(
        const ggml_backend_hrx_device_context * device_context,
        ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:
            return &device_context->mul_mat_vec_f16_batched_provider;
        case GGML_TYPE_F32:
            return &device_context->mul_mat_vec_f32_batched_provider;
        default:
            return nullptr;
    }
}

static bool ggml_backend_hrx_provider_available(const ggml_backend_hrx_op_provider & provider);

static int ggml_backend_hrx_mul_mat_vec_bf16_workgroup_size_from_env() {
    const uint64_t value = ggml_backend_hrx_u64_from_env("GGML_HRX_MUL_MAT_VEC_BF16_WORKGROUP_SIZE", 0);
    return value == 64 || value == 128 || value == 256 ? static_cast<int>(value) : 0;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_vec_bf16_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t k,
        int64_t rows,
        int64_t cols) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS2_COLS1_DECODE") &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS4_K2048_COLS1_DECODE") &&
        cols == 1 && k == 2048 && rows >= 4 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_rows4_k2048_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_rows4_k2048_cols1_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS2_COLS1_DECODE") &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS4_K512_COLS1_DECODE") &&
        cols == 1 && k == 512 && rows == 2048 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_rows4_k512_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_rows4_k512_cols1_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS2_COLS1_DECODE") &&
        ggml_backend_hrx_env_enabled("GGML_HRX_ENABLE_BF16_ROWS2_COLS1_WG32_DECODE") &&
        cols == 1 && k == 512 && rows == 2048 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_rows2_cols1_wg32_provider)) {
        return &device_context->mul_mat_vec_bf16_rows2_cols1_wg32_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS2_COLS1_DECODE") &&
        cols == 1 && rows >= 2 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_rows2_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_rows2_cols1_provider;
    }
    if (cols == 1 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_cols1_provider;
    }
    if (!ggml_backend_hrx_approximate_kernels_disabled() &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_WMMA16_PROMPT") &&
        (k % 16) == 0 && cols == 512 && (rows % 16) == 0 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_wmma16_provider)) {
        return &device_context->mul_mat_vec_bf16_wmma16_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_ROWS2_COLS16_PROMPT") &&
        cols == 512 && (rows % 2) == 0 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_rows2_cols16_provider)) {
        return &device_context->mul_mat_vec_bf16_rows2_cols16_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_COLS32_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_cols32_provider)) {
        return &device_context->mul_mat_vec_bf16_cols32_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_COLS16_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_cols16_provider)) {
        return &device_context->mul_mat_vec_bf16_cols16_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_COLS8_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_cols8_provider)) {
        return &device_context->mul_mat_vec_bf16_cols8_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_COLS4_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_cols4_provider)) {
        return &device_context->mul_mat_vec_bf16_cols4_provider;
    }

    const int workgroup_size = ggml_backend_hrx_mul_mat_vec_bf16_workgroup_size_from_env();
    const ggml_backend_hrx_op_provider * provider =
        workgroup_size == 64 ? &device_context->mul_mat_vec_bf16_wg64_provider :
        workgroup_size == 128 ? &device_context->mul_mat_vec_bf16_wg128_provider :
        &device_context->mul_mat_vec_bf16_provider;
    return ggml_backend_hrx_provider_available(*provider) ? provider : &device_context->mul_mat_vec_bf16_provider;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_vec_bf16_swiglu_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t k,
        int64_t rows,
        int64_t cols) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_ROWS4_K2048_COLS1_DECODE") &&
        cols == 1 && k == 2048 && rows >= 4 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider;
    }
    if (ggml_backend_hrx_env_enabled("GGML_HRX_ENABLE_BF16_SWIGLU_ROWS2_COLS1_DECODE") &&
        cols == 1 && rows >= 2 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_rows2_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_rows2_cols1_provider;
    }
    if (cols == 1 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_cols1_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_cols1_provider;
    }
    if (!ggml_backend_hrx_approximate_kernels_disabled() &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_WMMA16_PROMPT") &&
        (k % 16) == 0 && cols == 512 && (rows % 16) == 0 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_wmma16_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_wmma16_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_ROWS2_COLS8_PROMPT") &&
        cols == 512 && (rows % 2) == 0 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_COLS16_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_cols16_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_cols16_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_COLS8_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_cols8_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_cols8_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_BF16_SWIGLU_COLS4_PROMPT") &&
        cols == 512 && ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_swiglu_cols4_provider)) {
        return &device_context->mul_mat_vec_bf16_swiglu_cols4_provider;
    }

    const int workgroup_size = ggml_backend_hrx_mul_mat_vec_bf16_workgroup_size_from_env();
    const ggml_backend_hrx_op_provider * provider =
        workgroup_size == 64 ? &device_context->mul_mat_vec_bf16_swiglu_wg64_provider :
        workgroup_size == 128 ? &device_context->mul_mat_vec_bf16_swiglu_wg128_provider :
        &device_context->mul_mat_vec_bf16_swiglu_provider;
    return ggml_backend_hrx_provider_available(*provider) ? provider :
        &device_context->mul_mat_vec_bf16_swiglu_provider;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_vec_batched_provider(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (src0->type == GGML_TYPE_F16) {
        if (src1->ne[1] == 1 &&
            src0->ne[3] == 1 &&
            src1->ne[3] == 1 &&
            op->ne[3] == 1 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f16_batched_cols1_provider)) {
            return &device_context->mul_mat_vec_f16_batched_cols1_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F16_BATCHED_COLS16_PROMPT") &&
            src1->ne[1] == 512 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f16_batched_cols16_provider)) {
            return &device_context->mul_mat_vec_f16_batched_cols16_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F16_BATCHED_COLS8_PROMPT") &&
            src1->ne[1] == 512 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f16_batched_cols8_provider)) {
            return &device_context->mul_mat_vec_f16_batched_cols8_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F16_BATCHED_COLS4_PROMPT") &&
            src1->ne[1] == 512 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f16_batched_cols4_provider)) {
            return &device_context->mul_mat_vec_f16_batched_cols4_provider;
        }
        return &device_context->mul_mat_vec_f16_batched_provider;
    }
    if (src0->type == GGML_TYPE_F32) {
        if (src1->ne[1] == 1 && op->ne[2] == 1 && src0->ne[0] == 2048 &&
            ggml_backend_hrx_provider_available(
                device_context->mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_provider)) {
            return &device_context->mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_provider;
        }
        if (src1->ne[1] == 1 && op->ne[2] == 1 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f32_batched_cols1_ne2_1_provider)) {
            return &device_context->mul_mat_vec_f32_batched_cols1_ne2_1_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F32_BATCHED_ROWS2_COLS8_PROMPT") &&
            src1->ne[1] == 512 && (src0->ne[1] % 2) == 0 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f32_batched_rows2_cols8_provider)) {
            return &device_context->mul_mat_vec_f32_batched_rows2_cols8_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F32_BATCHED_COLS16_PROMPT") &&
            src1->ne[1] == 512 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f32_batched_cols16_provider)) {
            return &device_context->mul_mat_vec_f32_batched_cols16_provider;
        }
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_F32_BATCHED_COLS8_PROMPT") &&
            src1->ne[1] == 512 &&
            ggml_backend_hrx_provider_available(device_context->mul_mat_vec_f32_batched_cols8_provider)) {
            return &device_context->mul_mat_vec_f32_batched_cols8_provider;
        }
        return &device_context->mul_mat_vec_f32_batched_provider;
    }
    return nullptr;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_vec_q8_0_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t cols) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_0_COLS8_PROMPT") &&
        cols == 512 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_q8_0_cols8_provider)) {
        return &device_context->mul_mat_vec_q8_0_cols8_provider;
    }
    return &device_context->mul_mat_vec_q8_0_provider;
}

static bool ggml_backend_hrx_q8_1_mmvq_forced() {
    const char * value = std::getenv("GGML_HRX_Q8_1_MMVQ");
    return value && (std::strcmp(value, "all") == 0 || std::strcmp(value, "1") == 0);
}

static constexpr int64_t GGML_HRX_Q8_1_MMVQ_AUTO_K_MIN = 2048;
static constexpr int64_t GGML_HRX_Q8_1_MMVQ_AUTO_Q4_K_ROWS_MIN = 4096;
static constexpr int64_t GGML_HRX_Q8_1_MMVQ_AUTO_Q5_Q6_K_ROWS_MIN = 2048;
static constexpr int64_t GGML_HRX_Q8_1_MMVQ_AUTO_Q8_0_ROWS_MIN = 2048;

static bool ggml_backend_hrx_q8_1_mmvq_auto_shape(const ggml_tensor * op, ggml_type type) {
    const ggml_tensor * src0 = op->src[0];
    if (!src0 || src0->ne[0] < GGML_HRX_Q8_1_MMVQ_AUTO_K_MIN) {
        return false;
    }

    // Conservative auto-policy gates from W7900 profiling on our target
    // quantized Qwen model; do not treat them as broadly tuned performance
    // crossover points. Use GGML_HRX_Q8_1_MMVQ=all to force smaller
    // validation or differential-test shapes.
    switch (type) {
        case GGML_TYPE_Q4_K:
            return src0->ne[1] >= GGML_HRX_Q8_1_MMVQ_AUTO_Q4_K_ROWS_MIN;
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
            return src0->ne[1] >= GGML_HRX_Q8_1_MMVQ_AUTO_Q5_Q6_K_ROWS_MIN;
        case GGML_TYPE_Q8_0:
            return src0->ne[1] >= GGML_HRX_Q8_1_MMVQ_AUTO_Q8_0_ROWS_MIN;
        default:
            return false;
    }
}

static bool ggml_backend_hrx_supports_mul_mat_vec_2d(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    const ggml_backend_hrx_op_provider * provider = src0->type == GGML_TYPE_BF16 ?
        ggml_backend_hrx_select_mul_mat_vec_bf16_provider(device_context, src0->ne[0], src0->ne[1], src1->ne[1]) :
        src0->type == GGML_TYPE_Q8_0 ?
        ggml_backend_hrx_select_mul_mat_vec_q8_0_provider(device_context, src1->ne[1]) :
        ggml_backend_hrx_mul_mat_vec_provider(device_context, src0->type);
    const int64_t block_size = ggml_blck_size(src0->type);
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src1->ne[1] > 0 &&
           src0->ne[0] > 0 &&
           src0->ne[1] > 0 &&
           block_size > 0 &&
           (src0->ne[0] % block_size) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_mul_mat_vec_k_quant_q8_1_shape(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op,
        ggml_type type,
        const ggml_backend_hrx_op_provider & provider) {
    GGML_UNUSED(device_context);
    if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") ||
        (!ggml_backend_hrx_q8_1_mmvq_forced() && !ggml_backend_hrx_q8_1_mmvq_auto_shape(op, type)) ||
        provider.kind != ggml_backend_hrx_provider_kind::hsaco) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return src0 && src1 &&
           src0->type == type &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src1->ne[1] > 0 &&
           src0->ne[0] > 0 &&
           src0->ne[1] > 0 &&
           (src0->ne[0] % 256) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

struct ggml_backend_hrx_q8_1_mmvq_variant {
    const ggml_backend_hrx_op_provider * provider = nullptr;
    bool x4_quant = false;
    uint32_t rows_per_workgroup = 1;
    uint32_t cols_per_workgroup = 1;
};

static bool ggml_backend_hrx_provider_available(const ggml_backend_hrx_op_provider & provider) {
    return provider.kind == ggml_backend_hrx_provider_kind::hsaco;
}

static bool ggml_backend_hrx_supports_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op,
        const ggml_backend_hrx_op_provider & provider,
        const char * disable_env) {
    if (!op ||
        ggml_backend_hrx_approximate_kernels_disabled() ||
        ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") ||
        ggml_backend_hrx_env_enabled(disable_env) ||
        (!ggml_backend_hrx_q8_1_mmvq_forced() && !ggml_backend_hrx_q8_1_mmvq_auto_shape(op, GGML_TYPE_Q8_0)) ||
        !ggml_backend_hrx_provider_available(device_context->quantize_q8_1_x4_provider) ||
        !ggml_backend_hrx_provider_available(provider)) {
        return false;
    }

    const ggml_tensor * src0 = op ? op->src[0] : nullptr;
    const ggml_tensor * src1 = op ? op->src[1] : nullptr;
    return src0 &&
           src1 &&
           src0->type == GGML_TYPE_Q8_0 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           src0->ne[0] > 0 &&
           src0->ne[1] > 0 &&
           src1->ne[1] == 512 &&
           (src0->ne[0] % 128) == 0 &&
           (src0->ne[1] % 128) == 0 &&
           (src1->ne[1] % 32) == 0 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op);
}

static ggml_backend_hrx_q8_1_mmvq_variant ggml_backend_hrx_mul_mat_vec_k_q8_1_variant(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    ggml_backend_hrx_q8_1_mmvq_variant variant = {};
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return variant;
    }

    const bool has_q8_1 = ggml_backend_hrx_provider_available(device_context->quantize_q8_1_provider);
    const bool has_q8_1_x4 = ggml_backend_hrx_provider_available(device_context->quantize_q8_1_x4_provider);
    const int64_t rows = src0->ne[1];
    const int64_t cols = src1->ne[1];

    switch (src0->type) {
        case GGML_TYPE_Q4_K:
            if (has_q8_1 && ggml_backend_hrx_supports_mul_mat_vec_k_quant_q8_1_shape(
                    device_context, op, GGML_TYPE_Q4_K, device_context->mul_mat_vec_q4_k_q8_1_provider)) {
                variant.provider = &device_context->mul_mat_vec_q4_k_q8_1_provider;
            }
            return variant;
        case GGML_TYPE_Q5_K:
            if (!ggml_backend_hrx_supports_mul_mat_vec_k_quant_q8_1_shape(
                    device_context, op, GGML_TYPE_Q5_K, device_context->mul_mat_vec_q5_k_q8_1_provider)) {
                return variant;
            }
            if (has_q8_1_x4 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQL128") &&
                rows % 128 == 0 && cols % 128 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_provider)) {
                variant.provider = &device_context->mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 128;
                variant.cols_per_workgroup = 128;
                return variant;
            }
            if (has_q8_1_x4 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ64") &&
                rows % 64 == 0 && cols % 64 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_provider)) {
                variant.provider = &device_context->mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 64;
                variant.cols_per_workgroup = 64;
                return variant;
            }
            if (has_q8_1_x4 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q5_K_Q8_1_X4_MMQ32") &&
                rows % 32 == 0 && cols % 32 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_provider)) {
                variant.provider = &device_context->mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 32;
                variant.cols_per_workgroup = 32;
                return variant;
            }
            if (has_q8_1 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q5_K_Q8_1_MMQ32") &&
                rows % 32 == 0 && cols % 32 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_provider)) {
                variant.provider = &device_context->mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_provider;
                variant.rows_per_workgroup = 32;
                variant.cols_per_workgroup = 32;
                return variant;
            }
            if (has_q8_1) {
                variant.provider = &device_context->mul_mat_vec_q5_k_q8_1_provider;
            }
            return variant;
        case GGML_TYPE_Q6_K:
            if (!ggml_backend_hrx_supports_mul_mat_vec_k_quant_q8_1_shape(
                    device_context, op, GGML_TYPE_Q6_K, device_context->mul_mat_vec_q6_k_q8_1_provider)) {
                return variant;
            }
            if (has_q8_1_x4 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q6_K_Q8_1_X4_MMQL128") &&
                rows % 128 == 0 && cols % 64 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_provider)) {
                variant.provider = &device_context->mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 128;
                variant.cols_per_workgroup = 64;
                return variant;
            }
            if (has_q8_1_x4 &&
                !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q6_K_Q8_1_X4_MMQ32") &&
                rows % 32 == 0 && cols % 32 == 0 &&
                ggml_backend_hrx_provider_available(
                    device_context->mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_provider)) {
                variant.provider = &device_context->mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 32;
                variant.cols_per_workgroup = 32;
                return variant;
            }
            if (has_q8_1) {
                variant.provider = &device_context->mul_mat_vec_q6_k_q8_1_provider;
            }
            return variant;
        case GGML_TYPE_Q8_0:
            if (has_q8_1_x4 &&
                ggml_backend_hrx_supports_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt(
                    device_context,
                    op,
                    device_context->mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_provider,
                    "GGML_HRX_DISABLE_Q8_0_Q8_1_X4_MMQ128X32_PROMPT")) {
                variant.provider = &device_context->mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_provider;
                variant.x4_quant = true;
                variant.rows_per_workgroup = 128;
                variant.cols_per_workgroup = 32;
                return variant;
            }
            return variant;
        default:
            return variant;
    }
}

static bool ggml_backend_hrx_supports_mul_mat_vec_k_q8_1(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx_mul_mat_vec_k_q8_1_variant(device_context, op).provider != nullptr;
}

static bool ggml_backend_hrx_supports_mul_mat_vec_batched(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (!src0 || !src1) {
        return false;
    }

    const ggml_backend_hrx_op_provider * provider =
        ggml_backend_hrx_select_mul_mat_vec_batched_provider(device_context, op);
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           (src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_F32) &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src1->ne[1] &&
           op->ne[2] == src1->ne[2] &&
           op->ne[3] == src1->ne[3] &&
           (src0->ne[2] != 1 || src0->ne[3] != 1 ||
            src1->ne[2] != 1 || src1->ne[3] != 1 ||
            op->ne[2] != 1 || op->ne[3] != 1) &&
           src0->ne[0] > 0 &&
           src0->ne[1] > 0 &&
           src0->ne[2] > 0 &&
           src0->ne[3] > 0 &&
           src1->ne[1] > 0 &&
           src1->ne[2] > 0 &&
           src1->ne[3] > 0 &&
           src0->ne[1] <= std::numeric_limits<uint32_t>::max() &&
           src1->ne[1] <= std::numeric_limits<uint32_t>::max() / op->ne[2] &&
           src1->ne[1] * op->ne[2] <= std::numeric_limits<uint32_t>::max() / op->ne[3] &&
           op->ne[2] >= src0->ne[2] &&
           op->ne[3] >= src0->ne[3] &&
           (op->ne[2] % src0->ne[2]) == 0 &&
           (op->ne[3] % src0->ne[3]) == 0 &&
           src0->nb[0] == ggml_type_size(src0->type) &&
           src1->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_supports_mul_mat_vec(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_backend_hrx_approximate_kernels_disabled()) {
        return false;
    }

    return ggml_backend_hrx_supports_mul_mat_vec_2d(device_context, op) ||
           ggml_backend_hrx_supports_mul_mat_vec_k_q8_1(device_context, op) ||
           ggml_backend_hrx_supports_mul_mat_vec_batched(device_context, op);
}

static bool ggml_backend_hrx_supports_mul_mat_vec_q8_0_add(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mm,
        const ggml_tensor * add) {
    const bool has_scalar_provider =
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_q8_0_add_provider);
    const bool has_cols8_provider =
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_0_COLS8_PROMPT") &&
        mm && mm->src[1] && mm->src[1]->ne[1] == 512 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_q8_0_add_cols8_provider);
    const bool has_rows4_cols4_provider =
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_0_ADD_ROWS4_COLS4_PROMPT") &&
        mm && mm->src[0] && mm->src[1] &&
        (mm->src[0]->ne[1] % 4) == 0 &&
        mm->src[1]->ne[1] == 512 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_vec_q8_0_add_rows4_cols4_provider);
    const bool has_q8_1_x4_mmq128x32_provider =
        ggml_backend_hrx_supports_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt(
            device_context,
            mm,
            device_context->mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider,
            "GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT");
    if ((!has_scalar_provider && !has_cols8_provider && !has_rows4_cols4_provider &&
         !has_q8_1_x4_mmq128x32_provider) ||
        !mm ||
        !mm->src[0] ||
        mm->src[0]->type != GGML_TYPE_Q8_0 ||
        !ggml_backend_hrx_supports_mul_mat_vec(device_context, mm) ||
        !add ||
        add->op != GGML_OP_ADD ||
        add->type != GGML_TYPE_F32 ||
        (add->src[0] != mm && add->src[1] != mm) ||
        !ggml_are_same_shape(add, mm) ||
        !ggml_is_contiguous(add)) {
        return false;
    }

    const ggml_tensor * bias = add->src[0] == mm ? add->src[1] : add->src[0];
    return bias &&
           bias->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(bias, mm) &&
           ggml_is_contiguous(bias);
}

static bool ggml_backend_hrx_supports_mul_mat_vec_bf16_swiglu(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * swiglu) {
    if (!gate || !up || !swiglu ||
        gate->op != GGML_OP_MUL_MAT ||
        up->op != GGML_OP_MUL_MAT ||
        swiglu->op != GGML_OP_GLU ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(swiglu, 1) != 0 ||
        swiglu->src[0] != gate ||
        swiglu->src[1] != up ||
        !ggml_backend_hrx_supports_mul_mat_vec(device_context, gate) ||
        !ggml_backend_hrx_supports_mul_mat_vec(device_context, up)) {
        return false;
    }

    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_vec_bf16_swiglu_provider(
        device_context, up->src[0]->ne[0], up->src[0]->ne[1], up->src[1]->ne[1]);
    return provider &&
           ggml_backend_hrx_provider_available(*provider) &&
           gate->src[0]->type == GGML_TYPE_BF16 &&
           up->src[0]->type == GGML_TYPE_BF16 &&
           gate->src[1] == up->src[1] &&
           ggml_are_same_shape(gate->src[0], up->src[0]) &&
           ggml_are_same_stride(gate->src[0], up->src[0]) &&
           ggml_are_same_shape(gate, up) &&
           ggml_are_same_shape(swiglu, up) &&
           swiglu->type == GGML_TYPE_F32 &&
           swiglu->nb[0] == ggml_type_size(swiglu->type);
}

static bool ggml_backend_hrx_supports_mul_mat_vec_bf16_set_rows_f16(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * adapter,
        const ggml_tensor * set_rows) {
    return ggml_backend_hrx_provider_available(device_context->mul_mat_vec_bf16_set_rows_f16_provider) &&
           !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
           !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_MAT_SET_ROWS_FUSION") &&
           mul_mat &&
           adapter &&
           set_rows &&
           mul_mat->op == GGML_OP_MUL_MAT &&
           set_rows->op == GGML_OP_SET_ROWS &&
           ggml_backend_hrx_unwrap_reshape_view_src0(adapter) == mul_mat &&
           set_rows->src[0] == adapter &&
           set_rows->type == GGML_TYPE_F16 &&
           set_rows->src[1] &&
           set_rows->src[1]->type == GGML_TYPE_I64 &&
           set_rows->src[2] &&
           set_rows->src[2]->type == GGML_TYPE_F16 &&
           ggml_backend_hrx_supports_mul_mat_vec(device_context, mul_mat) &&
           ggml_backend_hrx_supports_set_rows(device_context, set_rows) &&
           mul_mat->src[0]->type == GGML_TYPE_BF16 &&
           mul_mat->src[1]->ne[1] == 1 &&
           adapter->ne[0] == 1 &&
           adapter->ne[1] == mul_mat->ne[0] &&
           adapter->ne[2] == 1 &&
           adapter->ne[3] == 1 &&
           set_rows->ne[0] == 1 &&
           set_rows->ne[2] == 1 &&
           set_rows->ne[3] == 1;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_id_q4_k_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_ID_Q8_1_X4_MMQ_PROMPT") &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") &&
        k == 512 && rows % 64 == 0 && n_ids == 8 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->clear_u32_provider) &&
        ggml_backend_hrx_provider_available(device_context->compact_moe_routes_provider) &&
        ggml_backend_hrx_provider_available(device_context->quantize_q8_1_x4_provider) &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_ID_GROUPED_PROMPT") &&
        k == 512 && rows % 2 == 0 && n_ids == 8 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->clear_u32_provider) &&
        ggml_backend_hrx_provider_available(device_context->compact_moe_routes_provider) &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_grouped_row2_route8_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_grouped_row2_route8_wg64_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_ID_ROW4_PROMPT") &&
        k == 512 && rows % 4 == 0 && n_ids > 0 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_row4_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_row4_wg64_provider;
    }
    if (k <= 2048 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_wg64_provider;
    }
    return &device_context->mul_mat_id_q4_k_provider;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_id_q4_k_mul_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_PACKED_Q4_K_MUL") &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_PACKED_Q4_K_MUL_ROWS2_X16") &&
        k == 512 && rows == 2048 && n_ids == 8 && n_tokens == 1 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_mul_rows2_x16_wg32_provider)) {
        return &device_context->mul_mat_id_q4_k_mul_rows2_x16_wg32_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_PACKED_Q4_K_MUL") &&
        k == 512 && rows == 2048 && n_ids == 8 && n_tokens == 1 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_mul_packed_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_mul_packed_wg64_provider;
    }
    if (k <= 2048 && ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_mul_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_mul_wg64_provider;
    }
    return &device_context->mul_mat_id_q4_k_mul_provider;
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_select_mul_mat_id_q4_k_swiglu_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t k,
        int64_t rows,
        int64_t n_ids,
        int64_t n_tokens) {
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_SWIGLU_Q8_1_X4_MMQ_PROMPT") &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") &&
        k == 2048 && rows % 16 == 0 && n_ids == 8 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->clear_u32_provider) &&
        ggml_backend_hrx_provider_available(device_context->compact_moe_routes_provider) &&
        ggml_backend_hrx_provider_available(device_context->quantize_q8_1_x4_provider) &&
        ggml_backend_hrx_provider_available(
            device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_PACKED_Q4_K_SWIGLU") &&
        k == 2048 && rows == 512 && n_ids == 8 && n_tokens == 1 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_swiglu_packed_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_swiglu_packed_wg64_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_SWIGLU_GROUPED_PROMPT") &&
        k == 2048 && rows % 2 == 0 && n_ids == 8 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->clear_u32_provider) &&
        ggml_backend_hrx_provider_available(device_context->compact_moe_routes_provider) &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider;
    }
    if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q4_K_SWIGLU_ROW4_PROMPT") &&
        k == 2048 && rows % 4 == 0 && n_ids == 8 && n_tokens == 512 &&
        ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_swiglu_row4_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_swiglu_row4_wg64_provider;
    }
    if (k <= 2048 && ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_swiglu_wg64_provider)) {
        return &device_context->mul_mat_id_q4_k_swiglu_wg64_provider;
    }
    return &device_context->mul_mat_id_q4_k_swiglu_provider;
}

static bool ggml_backend_hrx_supports_mul_mat_id_q4_k(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_backend_hrx_approximate_kernels_disabled()) {
        return false;
    }

    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    if (!src0 || !src1 || !src2) {
        return false;
    }

    const ggml_backend_hrx_op_provider * provider =
        ggml_backend_hrx_select_mul_mat_id_q4_k_provider(device_context, src0->ne[0], src0->ne[1], src2->ne[0], src2->ne[1]);
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0->type == GGML_TYPE_Q4_K &&
           src1->type == GGML_TYPE_F32 &&
           src2->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] > 0 &&
           src0->ne[1] > 0 &&
           src0->ne[2] > 0 &&
           src1->ne[1] > 0 &&
           src1->ne[2] > 0 &&
           src2->ne[0] > 0 &&
           src2->ne[1] > 0 &&
           src0->ne[0] % 256 == 0 &&
           src0->ne[3] == 1 &&
           src1->ne[3] == 1 &&
           src2->ne[2] == 1 &&
           src2->ne[3] == 1 &&
           src0->ne[0] == src1->ne[0] &&
           src2->ne[1] == src1->ne[2] &&
           (src2->ne[0] == src1->ne[1] || src1->ne[1] == 1) &&
           op->ne[0] == src0->ne[1] &&
           op->ne[1] == src2->ne[0] &&
           op->ne[2] == src2->ne[1] &&
           op->ne[3] == 1 &&
           src0->ne[1] <= std::numeric_limits<uint32_t>::max() &&
           src2->ne[0] <= std::numeric_limits<uint32_t>::max() / src2->ne[1] &&
           src0->nb[0] == ggml_type_size(src0->type) &&
           src1->nb[0] == ggml_type_size(src1->type) &&
           src2->nb[0] == ggml_type_size(src2->type) &&
           op->nb[0] == ggml_type_size(op->type);
}

static bool ggml_backend_hrx_supports_mul_mat_id_q4_k_q8_1(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") &&
           ggml_backend_hrx_q8_1_mmvq_forced() &&
           ggml_backend_hrx_provider_available(device_context->quantize_q8_1_provider) &&
           ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_q8_1_provider) &&
           ggml_backend_hrx_supports_mul_mat_id_q4_k(device_context, op);
}

static bool ggml_backend_hrx_supports_mul_mat_id_q4_k_mul(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    if (!mmid || !mul ||
        !ggml_backend_hrx_supports_mul_mat_id_q4_k(device_context, mmid) ||
        mul->op != GGML_OP_MUL ||
        mul->type != GGML_TYPE_F32 ||
        mul->src[0] != mmid ||
        !mul->src[1] ||
        mul->src[1]->type != GGML_TYPE_F32 ||
        !ggml_are_same_shape(mul, mmid) ||
        mul->nb[0] != ggml_type_size(mul->type)) {
        return false;
    }

    const ggml_tensor * scale = mul->src[1];
    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_id_q4_k_mul_provider(
        device_context, mmid->src[0]->ne[0], mmid->src[0]->ne[1], mmid->src[2]->ne[0], mmid->src[2]->ne[1]);
    return provider &&
           ggml_backend_hrx_provider_available(*provider) &&
           scale->ne[0] == 1 &&
           scale->ne[1] == mmid->ne[1] &&
           scale->ne[2] == 1 &&
           scale->ne[3] == 1 &&
           scale->nb[0] == ggml_type_size(scale->type) &&
           ggml_is_contiguous(scale);
}

static bool ggml_backend_hrx_supports_mul_mat_id_q4_k_mul_q8_1(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_1_MMVQ") &&
           ggml_backend_hrx_q8_1_mmvq_forced() &&
           ggml_backend_hrx_provider_available(device_context->quantize_q8_1_provider) &&
           ggml_backend_hrx_provider_available(device_context->mul_mat_id_q4_k_mul_q8_1_provider) &&
           ggml_backend_hrx_supports_mul_mat_id_q4_k_mul(device_context, mmid, mul);
}

static bool ggml_backend_hrx_supports_mul_mat_id_q4_k_swiglu(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * swiglu) {
    if (!gate || !up || !swiglu ||
        gate->op != GGML_OP_MUL_MAT_ID ||
        up->op != GGML_OP_MUL_MAT_ID ||
        swiglu->op != GGML_OP_GLU ||
        ggml_get_glu_op(swiglu) != GGML_GLU_OP_SWIGLU ||
        ggml_get_op_params_i32(swiglu, 1) != 0 ||
        swiglu->src[0] != gate ||
        swiglu->src[1] != up ||
        !ggml_backend_hrx_supports_mul_mat_id_q4_k(device_context, gate) ||
        !ggml_backend_hrx_supports_mul_mat_id_q4_k(device_context, up)) {
        return false;
    }

    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_id_q4_k_swiglu_provider(
        device_context, up->src[0]->ne[0], up->src[0]->ne[1], up->src[2]->ne[0], up->src[2]->ne[1]);
    return provider &&
           ggml_backend_hrx_provider_available(*provider) &&
           gate->src[1] == up->src[1] &&
           gate->src[2] == up->src[2] &&
           gate->src[0]->type == up->src[0]->type &&
           ggml_are_same_shape(gate->src[0], up->src[0]) &&
           ggml_are_same_stride(gate->src[0], up->src[0]) &&
           ggml_are_same_shape(gate, up) &&
           ggml_are_same_shape(swiglu, up) &&
           swiglu->type == GGML_TYPE_F32 &&
           swiglu->nb[0] == ggml_type_size(swiglu->type);
}

static const ggml_backend_hrx_op_provider * ggml_backend_hrx_flash_attn_ext_f32_decode_provider(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * k,
        const ggml_tensor * v) {
    if (ggml_backend_hrx_flash_attn_ext_decode_disabled()) {
        return nullptr;
    }
    if (k->type == GGML_TYPE_F16 && v->type == GGML_TYPE_F16) {
        return &device_context->flash_attn_ext_f16_provider;
    }
    if (k->type == GGML_TYPE_BF16 && v->type == GGML_TYPE_BF16) {
        return &device_context->flash_attn_ext_bf16_provider;
    }
    if (k->type == GGML_TYPE_F32 && v->type == GGML_TYPE_F32) {
        return &device_context->flash_attn_ext_f32_provider;
    }
    if (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q8_0) {
        return &device_context->flash_attn_ext_q8_0_provider;
    }
    if (k->type == GGML_TYPE_Q8_0 && v->type == GGML_TYPE_Q4_0) {
        return &device_context->flash_attn_ext_q8_0_q4_0_provider;
    }
    if (k->type == GGML_TYPE_Q4_0 && v->type == GGML_TYPE_Q4_0) {
        return &device_context->flash_attn_ext_q4_0_provider;
    }
    return nullptr;
}

static bool ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_common(
        const ggml_tensor * op,
        const ggml_backend_hrx_op_provider & provider,
        const char * disable_knob,
        bool require_no_bias_or_softcap) {
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        ggml_backend_hrx_env_enabled(disable_knob) ||
        provider.kind != ggml_backend_hrx_provider_kind::hsaco) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    const ggml_tensor * sinks = op->src[4];
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(op->op_params) + 1, sizeof(float));
    std::memcpy(&logit_softcap, reinterpret_cast<const int32_t *>(op->op_params) + 2, sizeof(float));
    if (require_no_bias_or_softcap && (max_bias != 0.0f || logit_softcap != 0.0f)) {
        return false;
    }

    return q && k && v && mask &&
           !sinks &&
           q->type == GGML_TYPE_F32 &&
           k->type == GGML_TYPE_F16 &&
           v->type == GGML_TYPE_F16 &&
           mask->type == GGML_TYPE_F16 &&
           op->type == GGML_TYPE_F32 &&
           q->ne[0] == 256 &&
           k->ne[0] == 256 &&
           v->ne[0] == 256 &&
           op->ne[0] == 256 &&
           q->ne[1] == 512 &&
           k->ne[1] == 512 &&
           v->ne[1] == 512 &&
           q->ne[2] == 16 &&
           k->ne[2] == 2 &&
           v->ne[2] == 2 &&
           q->ne[3] == k->ne[3] &&
           q->ne[3] == v->ne[3] &&
           q->ne[2] == op->ne[1] &&
           q->ne[1] == op->ne[2] &&
           q->ne[3] == op->ne[3] &&
           mask->ne[0] == 512 &&
           mask->ne[1] >= 512 &&
           mask->ne[2] == 1 &&
           mask->ne[3] == q->ne[3] &&
           q->nb[0] == sizeof(float) &&
           k->nb[0] == ggml_type_size(k->type) &&
           v->nb[0] == ggml_type_size(v->type) &&
           mask->nb[0] == ggml_type_size(mask->type) &&
           op->nb[0] == sizeof(float) &&
           ggml_is_contiguous(mask) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_tile(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_common(
        op, device_context->flash_attn_ext_f16_prefill_tile_provider,
        "GGML_HRX_DISABLE_F16_PREFILL_FA_TILE", false);
}

static bool ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_wmma(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_common(
        op, device_context->flash_attn_ext_f16_prefill_wmma_provider,
        "GGML_HRX_DISABLE_F16_PREFILL_FA_WMMA", true);
}

static bool ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_direct(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    return ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_common(
        op, device_context->flash_attn_ext_f16_prefill_direct_provider,
        "GGML_HRX_DISABLE_F16_PREFILL_FA_DIRECT", true);
}

static bool ggml_backend_hrx_supports_flash_attn_ext_f32_decode(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    if (ggml_backend_hrx_approximate_kernels_disabled()) {
        return false;
    }

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * mask = op->src[3];
    const ggml_tensor * sinks = op->src[4];
    if (!q || !k || !v) {
        return false;
    }

    if (ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_direct(device_context, op) ||
        ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_wmma(device_context, op) ||
        ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_tile(device_context, op)) {
        return true;
    }

    const ggml_backend_hrx_op_provider * provider =
        ggml_backend_hrx_flash_attn_ext_f32_decode_provider(device_context, k, v);
    float max_bias = 0.0f;
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(op->op_params) + 1, sizeof(float));
    const bool permuted_q = q->nb[1] > q->nb[2];
    return provider &&
           provider->kind == ggml_backend_hrx_provider_kind::hsaco &&
           q->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           (!mask || mask->type == GGML_TYPE_F16) &&
           (!sinks || (sinks->type == GGML_TYPE_F32 &&
                       sinks->ne[0] == q->ne[2] &&
                       ggml_is_contiguous(sinks))) &&
           (max_bias == 0.0f || mask) &&
           q->ne[0] == k->ne[0] &&
           q->ne[0] == v->ne[0] &&
           q->ne[0] == op->ne[0] &&
           q->ne[0] <= 256 &&
           q->ne[1] <= 1024 &&
           k->ne[1] == v->ne[1] &&
           k->ne[1] <= 1024 &&
           k->ne[2] == v->ne[2] &&
           q->ne[3] == k->ne[3] &&
           q->ne[3] == v->ne[3] &&
           q->ne[2] == op->ne[1] &&
           q->ne[2] % k->ne[2] == 0 &&
           q->ne[1] == op->ne[2] &&
           q->ne[3] == op->ne[3] &&
           (k->type != GGML_TYPE_F32 || !permuted_q || q->ne[3] == 1 || q->ne[2] == k->ne[2]) &&
           q->nb[0] == sizeof(float) &&
           k->nb[0] == ggml_type_size(k->type) &&
           v->nb[0] == ggml_type_size(v->type) &&
           op->nb[0] == sizeof(float) &&
           (!mask ||
            (mask->ne[0] == k->ne[1] &&
             mask->ne[1] >= q->ne[1] &&
             mask->ne[2] == 1 &&
             mask->ne[3] == q->ne[3] &&
             mask->nb[0] == ggml_type_size(mask->type) &&
             ggml_is_contiguous(mask))) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_concat_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const int32_t dim = ggml_get_op_params_i32(op, 0);
    return device_context->concat_f32_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           dim == 0 &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[1] == src1->ne[1] &&
           src0->ne[1] == op->ne[1] &&
           src0->ne[2] == 1 && src0->ne[3] == 1 &&
           src1->ne[2] == 1 && src1->ne[3] == 1 &&
           op->ne[2] == 1 && op->ne[3] == 1 &&
           op->ne[0] == src0->ne[0] + src1->ne[0] &&
           op->nb[0] == sizeof(float);
}

static int32_t ggml_backend_hrx_next_power_of_2(int64_t value) {
    int32_t result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

static bool ggml_backend_hrx_supports_soft_max_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    float max_bias = 0.0f;
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(op->op_params) + 1, sizeof(float));
    const ggml_backend_hrx_op_provider & provider =
        src1 ? device_context->soft_max_f32_mask_provider : device_context->soft_max_f32_provider;
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           !src2 &&
           max_bias == 0.0f &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] > 0 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op) &&
           (!src1 ||
            (src1->type == GGML_TYPE_F32 &&
             src1->ne[0] == src0->ne[0] &&
             src1->ne[1] >= src0->ne[1] &&
             src1->ne[2] > 0 &&
             src1->ne[3] > 0 &&
             (src0->ne[2] % src1->ne[2]) == 0 &&
             (src0->ne[3] % src1->ne[3]) == 0 &&
             ggml_is_contiguous(src1)));
}

static bool ggml_backend_hrx_supports_argsort_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const int64_t ncols = src0 ? src0->ne[0] : 0;
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ARGSORT") &&
           device_context->argsort_f32_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src0->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_I32 &&
           ncols > 0 &&
           ncols <= 256 &&
           ggml_are_same_shape(src0, op) &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(op);
}

static bool ggml_backend_hrx_supports_topk_moe_output_layout(
        const ggml_tensor * tensor,
        int64_t n_expert_used) {
    return tensor &&
           (tensor->ne[0] == n_expert_used ||
            (tensor->ne[0] == 1 && tensor->ne[1] == n_expert_used));
}

static int64_t ggml_backend_hrx_topk_moe_row_stride(
        const ggml_tensor * tensor,
        int64_t n_expert_used) {
    if (tensor->ne[0] == n_expert_used) {
        return static_cast<int64_t>(tensor->nb[1]);
    }
    return static_cast<int64_t>(tensor->nb[2]);
}

static int64_t ggml_backend_hrx_topk_moe_k_stride(
        const ggml_tensor * tensor,
        int64_t n_expert_used) {
    if (tensor->ne[0] == n_expert_used) {
        return static_cast<int64_t>(tensor->nb[0]);
    }
    return static_cast<int64_t>(tensor->nb[1]);
}

static bool ggml_backend_hrx_supports_topk_moe_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * soft_max,
        const ggml_tensor * weights,
        const ggml_tensor * ids) {
    const int64_t n_logit_rows =
        (soft_max && soft_max->src[0]) ? ggml_nrows(soft_max->src[0]) : 0;
    const int64_t n_expert_used =
        (n_logit_rows > 0 && weights) ? ggml_nelements(weights) / n_logit_rows : 0;
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_TOPK_MOE") ||
        device_context->topk_moe_f32_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !soft_max || !weights || !ids ||
        soft_max->op != GGML_OP_SOFT_MAX ||
        !soft_max->src[0] ||
        soft_max->src[0]->type != GGML_TYPE_F32 ||
        soft_max->type != GGML_TYPE_F32 ||
        weights->type != GGML_TYPE_F32 ||
        ids->type != GGML_TYPE_I32 ||
        soft_max->src[1] || soft_max->src[2] ||
        soft_max->src[0]->ne[0] <= 0 ||
        soft_max->src[0]->ne[0] > 256 ||
        (soft_max->src[0]->ne[0] & (soft_max->src[0]->ne[0] - 1)) != 0 ||
        n_logit_rows <= 0 ||
        (n_logit_rows != 1 && !ggml_backend_hrx_env_enabled("GGML_HRX_ENABLE_PROMPT_TOPK_MOE")) ||
        ggml_nelements(weights) % n_logit_rows != 0 ||
        ggml_nelements(weights) != ggml_nelements(ids) ||
        n_expert_used <= 0 ||
        n_expert_used > 32 ||
        !ggml_backend_hrx_supports_topk_moe_output_layout(weights, n_expert_used) ||
        !ggml_backend_hrx_supports_topk_moe_output_layout(ids, n_expert_used) ||
        !ggml_is_contiguous(soft_max->src[0]) ||
        weights->nb[0] != sizeof(float) ||
        ids->nb[0] != sizeof(int32_t)) {
        return false;
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    std::memcpy(&scale, reinterpret_cast<const int32_t *>(soft_max->op_params), sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(soft_max->op_params) + 1, sizeof(float));
    return scale == 1.0f && max_bias == 0.0f;
}

static const ggml_backend_hrx_op_provider & ggml_backend_hrx_select_topk_moe_f32_provider(
        const ggml_backend_hrx_device_context * device_context,
        int64_t n_rows) {
    const auto provider_available = [](const ggml_backend_hrx_op_provider & provider) {
        return provider.kind == ggml_backend_hrx_provider_kind::hsaco;
    };

    const auto & baseline = device_context->topk_moe_f32_provider;
    if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_TOPK_SUBGROUP")) {
        return baseline;
    }

    switch (ggml_backend_hrx_topk_moe_variant_from_env()) {
        case ggml_backend_hrx_topk_moe_variant::baseline:
            return baseline;
        case ggml_backend_hrx_topk_moe_variant::shared4:
            if (provider_available(device_context->topk_moe_f32_shared4_provider)) {
                return device_context->topk_moe_f32_shared4_provider;
            }
            break;
        case ggml_backend_hrx_topk_moe_variant::wave32:
            if (provider_available(device_context->topk_moe_f32_wave32_provider)) {
                return device_context->topk_moe_f32_wave32_provider;
            }
            break;
        case ggml_backend_hrx_topk_moe_variant::auto_select:
            if (n_rows == 1 && provider_available(device_context->topk_moe_f32_wave32_provider)) {
                return device_context->topk_moe_f32_wave32_provider;
            }
            if (provider_available(device_context->topk_moe_f32_shared4_provider)) {
                return device_context->topk_moe_f32_shared4_provider;
            }
            break;
    }

    return baseline;
}

static bool ggml_backend_hrx_supports_rope_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    const ggml_tensor * src2 = op->src[2];
    float ext_factor = 0.0f;
    std::memcpy(&ext_factor, reinterpret_cast<const int32_t *>(op->op_params) + 7, sizeof(float));
    const int32_t n_dims = ggml_get_op_params_i32(op, 1);
    const int32_t mode = ggml_get_op_params_i32(op, 2);
    const int32_t section0 = ggml_get_op_params_i32(op, 11);
    const int32_t section1 = ggml_get_op_params_i32(op, 12);
    const int32_t section2 = ggml_get_op_params_i32(op, 13);
    const int32_t section3 = ggml_get_op_params_i32(op, 14);
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           device_context->rope_f32_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 &&
           src1 &&
           !src2 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_I32 &&
           op->type == GGML_TYPE_F32 &&
           mode == GGML_ROPE_TYPE_IMROPE &&
           ext_factor == 0.0f &&
           n_dims > 0 &&
           n_dims <= src0->ne[0] &&
           (n_dims % 2) == 0 &&
           (src0->ne[0] % 2) == 0 &&
           section0 + section1 + section2 + section3 > 0 &&
           src1->ne[0] == src0->ne[2] * 4 &&
           ggml_is_contiguous(src0) &&
           ggml_is_contiguous(src1) &&
           ggml_is_contiguous(op) &&
           ggml_are_same_shape(src0, op);
}

static bool ggml_backend_hrx_supports_rope_set_rows_f32_f16(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * rope,
        const ggml_tensor * view,
        const ggml_tensor * set_rows) {
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ROPE_SET_ROWS_FUSION") &&
           device_context->rope_set_rows_f32_f16_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           ggml_backend_hrx_supports_rope_f32(device_context, rope) &&
           view &&
           set_rows &&
           view->op == GGML_OP_VIEW &&
           set_rows->op == GGML_OP_SET_ROWS &&
           view->src[0] == rope &&
           set_rows->src[0] == view &&
           view->type == GGML_TYPE_F32 &&
           set_rows->type == GGML_TYPE_F16 &&
           rope->src[0]->ne[3] == 1 &&
           view->ne[0] == rope->ne[0] * rope->ne[1] &&
           ggml_is_contiguous(view) &&
           ggml_backend_hrx_supports_set_rows(device_context, set_rows);
}

static bool ggml_backend_hrx_supports_rms_norm_mul_rope_f32(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        const ggml_tensor * rope) {
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_RMS_NORM_MUL_ROPE_FUSION") &&
           device_context->rms_norm_mul_rope_f32_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           ggml_backend_hrx_supports_rms_norm_mul(device_context, rms_norm, mul) &&
           rope &&
           rope->op == GGML_OP_ROPE &&
           rope->src[0] == mul &&
           ggml_backend_hrx_supports_rope_f32(device_context, rope) &&
           rope->src[0]->ne[0] <= 1024;
}

static bool ggml_backend_hrx_supports_rms_norm_mul_rope_set_rows_f32_f16(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        const ggml_tensor * rope,
        const ggml_tensor * view,
        const ggml_tensor * set_rows) {
    return !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_RMS_NORM_MUL_ROPE_FUSION") &&
           device_context->rms_norm_mul_rope_set_rows_f32_f16_provider.kind ==
               ggml_backend_hrx_provider_kind::hsaco &&
           ggml_backend_hrx_supports_rms_norm_mul(device_context, rms_norm, mul) &&
           rope &&
           rope->op == GGML_OP_ROPE &&
           rope->src[0] == mul &&
           ggml_backend_hrx_supports_rope_set_rows_f32_f16(device_context, rope, view, set_rows) &&
           rope->src[0]->ne[0] <= 1024;
}

static bool ggml_backend_hrx_supports_ssm_conv(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    return device_context->ssm_conv_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           src0 && src1 &&
           src0->type == GGML_TYPE_F32 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == src1->ne[0] - 1 + op->ne[1] &&
           src0->ne[1] == src1->ne[1] &&
           src0->ne[2] == op->ne[2] &&
           src0->ne[3] == 1 &&
           src1->ne[2] == 1 &&
           src1->ne[3] == 1 &&
           op->ne[0] == src1->ne[1] &&
           op->ne[3] == 1 &&
           src0->nb[0] == sizeof(float) &&
           src1->nb[0] == sizeof(float) &&
           op->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_supports_ssm_conv_silu(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * ssm,
        const ggml_tensor * silu) {
    return !ggml_backend_hrx_approximate_kernels_disabled() &&
           device_context->silu_provider.kind == ggml_backend_hrx_provider_kind::hsaco &&
           silu &&
           silu->op == GGML_OP_UNARY &&
           ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU &&
           silu->src[0] == ssm &&
           silu->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(silu, ssm) &&
           silu->nb[0] == sizeof(float);
}

static bool ggml_backend_hrx_supports_ssm_conv_update(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * concat,
        const ggml_tensor * state_view,
        const ggml_tensor * state_update,
        const ggml_tensor * ssm,
        const ggml_tensor * silu) {
    if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_SSM_CONV_UPDATE_FUSION") ||
        device_context->ssm_conv_update_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !ggml_backend_hrx_supports_concat_f32(device_context, concat) ||
        !state_view ||
        state_view->op != GGML_OP_VIEW ||
        state_view->src[0] != concat ||
        state_view->view_src != concat ||
        !state_update ||
        state_update->op != GGML_OP_CPY ||
        state_update->src[0] != state_view ||
        !state_update->src[1] ||
        state_update->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(state_update) ||
        !ssm ||
        !ggml_backend_hrx_supports_ssm_conv(device_context, ssm) ||
        ssm->src[0] != concat) {
        return false;
    }

    const ggml_tensor * conv_state = concat->src[0];
    const ggml_tensor * input = concat->src[1];
    const ggml_tensor * weight = ssm->src[1];
    const int64_t conv_state_width = conv_state->ne[0];
    const int64_t n_tokens = input->ne[0];
    if (conv_state_width + 1 != weight->ne[0] ||
        n_tokens != ssm->ne[1] ||
        input->ne[1] != conv_state->ne[1] ||
        input->ne[2] != 1 ||
        input->ne[3] != 1 ||
        conv_state->ne[2] != 1 ||
        conv_state->ne[3] != 1 ||
        ssm->ne[2] != 1 ||
        state_view->ne[0] != conv_state_width ||
        state_view->ne[1] != conv_state->ne[1] ||
        state_view->ne[2] != 1 ||
        state_view->ne[3] != 1 ||
        state_view->nb[0] != sizeof(float) ||
        state_view->nb[1] != concat->nb[1] ||
        state_view->view_offs != static_cast<size_t>(n_tokens) * sizeof(float) ||
        ggml_nbytes(state_update) != static_cast<size_t>(conv_state_width * conv_state->ne[1]) * sizeof(float) ||
        ggml_backend_hrx_tensors_overlap(state_update, conv_state) ||
        ggml_backend_hrx_tensors_overlap(state_update, input)) {
        return false;
    }

    if (silu) {
        return ggml_backend_hrx_supports_ssm_conv_silu(device_context, ssm, silu);
    }
    return true;
}

static bool ggml_backend_hrx_supports_gated_delta_net(
        const ggml_backend_hrx_device_context * device_context,
        const ggml_tensor * op) {
    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * beta = op->src[4];
    const ggml_tensor * state = op->src[5];
    if (ggml_backend_hrx_approximate_kernels_disabled() ||
        device_context->gated_delta_net_provider.kind != ggml_backend_hrx_provider_kind::hsaco ||
        !q || !k || !v || !g || !beta || !state) {
        return false;
    }

    const int64_t S_v = v->ne[0];
    const int64_t H = v->ne[1];
    const int64_t n_tokens = v->ne[2];
    const int64_t n_seqs = v->ne[3];
    return q->type == GGML_TYPE_F32 &&
           k->type == GGML_TYPE_F32 &&
           v->type == GGML_TYPE_F32 &&
           g->type == GGML_TYPE_F32 &&
           beta->type == GGML_TYPE_F32 &&
           state->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           S_v > 0 && S_v <= 256 &&
           H > 0 &&
           n_tokens > 0 &&
           n_seqs > 0 &&
           q->ne[0] == S_v &&
           k->ne[0] == S_v &&
           q->ne[1] == k->ne[1] &&
           q->ne[1] > 0 &&
           H % q->ne[1] == 0 &&
           q->ne[2] == n_tokens &&
           k->ne[2] == n_tokens &&
           q->ne[3] > 0 &&
           k->ne[3] > 0 &&
           n_seqs % q->ne[3] == 0 &&
           n_seqs % k->ne[3] == 0 &&
           (g->ne[0] == 1 || g->ne[0] == S_v) &&
           g->ne[1] == H &&
           g->ne[2] == n_tokens &&
           g->ne[3] == n_seqs &&
           beta->ne[0] == 1 &&
           beta->ne[1] == H &&
           beta->ne[2] == n_tokens &&
           beta->ne[3] == n_seqs &&
           ggml_nelements(state) == S_v * S_v * H * n_seqs &&
           ggml_nelements(op) == S_v * H * n_tokens * n_seqs + S_v * S_v * H * n_seqs &&
           ggml_is_contiguous_rows(q) &&
           ggml_is_contiguous_rows(k) &&
           ggml_is_contiguous_rows(v) &&
           ggml_is_contiguous(g) &&
           ggml_is_contiguous(beta) &&
           ggml_is_contiguous(state) &&
           ggml_is_contiguous(op);
}

static ggml_status ggml_backend_hrx_dispatch_binary_elementwise(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst,
        const ggml_backend_hrx_op_provider & provider,
        const char * op_name) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: %s tensor is not backed by a HRX buffer\n", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_elementwise_constants constants = {
        /* .n = */ ggml_nelements(dst),
    };
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_broadcast_elementwise(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst,
        const ggml_backend_hrx_op_provider & provider,
        const char * op_name,
        bool linear_grid) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: broadcast %s tensor is not backed by a HRX buffer\n", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_broadcast_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .nrows    = */ ggml_nrows(dst),
        /* .ne1      = */ dst->ne[1],
        /* .ne2      = */ dst->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    const int64_t linear_count = constants.ne0 * constants.nrows;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(((linear_grid ? linear_count : constants.ne0) + workgroup_size - 1) / workgroup_size),
            static_cast<uint32_t>(linear_grid ? 1 : constants.nrows),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_mul_add_add_broadcast_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mul,
        const ggml_tensor * second_add,
        const ggml_tensor * add_src0,
        const ggml_tensor * add_src1) {
    const ggml_tensor * src0 = mul->src[0];
    const ggml_tensor * src1 = mul->src[1];
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(add_src0, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(add_src1, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(second_add, &bindings[4])) {
        GGML_LOG_ERROR("%s: fused MUL_ADD_ADD tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_add_add_broadcast_constants constants = {
        /* .ne0      = */ second_add->ne[0],
        /* .nrows    = */ ggml_nrows(second_add),
        /* .ne1      = */ second_add->ne[1],
        /* .ne2      = */ second_add->ne[2],
        /* .src1_ne0 = */ src1->ne[0],
        /* .src2_ne0 = */ add_src0->ne[0],
        /* .src3_ne0 = */ add_src1->ne[0],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .src1_nb1 = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2 = */ src1->ne[2] == 1 ? 0 : static_cast<int64_t>(src1->nb[2]),
        /* .src1_nb3 = */ src1->ne[3] == 1 ? 0 : static_cast<int64_t>(src1->nb[3]),
        /* .src2_nb1 = */ add_src0->ne[1] == 1 ? 0 : static_cast<int64_t>(add_src0->nb[1]),
        /* .src2_nb2 = */ add_src0->ne[2] == 1 ? 0 : static_cast<int64_t>(add_src0->nb[2]),
        /* .src2_nb3 = */ add_src0->ne[3] == 1 ? 0 : static_cast<int64_t>(add_src0->nb[3]),
        /* .src3_nb1 = */ add_src1->ne[1] == 1 ? 0 : static_cast<int64_t>(add_src1->nb[1]),
        /* .src3_nb2 = */ add_src1->ne[2] == 1 ? 0 : static_cast<int64_t>(add_src1->nb[2]),
        /* .src3_nb3 = */ add_src1->ne[3] == 1 ? 0 : static_cast<int64_t>(add_src1->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(second_add->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(second_add->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(second_add->nb[3]),
    };

    const auto & provider = context->device_context->mul_add_add_broadcast_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.ne0 + workgroup_size - 1) / workgroup_size),
            static_cast<uint32_t>(constants.nrows),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_add8_f32(
        ggml_backend_hrx_context * context,
        const std::array<const ggml_tensor *, 8> & sources,
        const ggml_tensor * dst) {
    hrx_buffer_ref_t bindings[9] = {};
    for (int i = 0; i < 8; ++i) {
        if (!ggml_backend_hrx_tensor_buffer_ref(sources[i], &bindings[i])) {
            GGML_LOG_ERROR("%s: ADD8 source tensor is not backed by a HRX buffer\n", __func__);
            return GGML_STATUS_FAILED;
        }
    }
    if (!ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[8])) {
        GGML_LOG_ERROR("%s: ADD8 destination tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_add8_constants constants = {
        /* .ncols    = */ dst->ne[0],
        /* .nrows    = */ ggml_nrows(dst),
        /* .src0_nb1 = */ static_cast<int64_t>(sources[0]->nb[1]),
        /* .src1_nb1 = */ static_cast<int64_t>(sources[1]->nb[1]),
        /* .src2_nb1 = */ static_cast<int64_t>(sources[2]->nb[1]),
        /* .src3_nb1 = */ static_cast<int64_t>(sources[3]->nb[1]),
        /* .src4_nb1 = */ static_cast<int64_t>(sources[4]->nb[1]),
        /* .src5_nb1 = */ static_cast<int64_t>(sources[5]->nb[1]),
        /* .src6_nb1 = */ static_cast<int64_t>(sources[6]->nb[1]),
        /* .src7_nb1 = */ static_cast<int64_t>(sources[7]->nb[1]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
    };

    const auto & provider = context->device_context->add8_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(((constants.ncols * constants.nrows) + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            9,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_mul_sum8_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mul,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = mul->src[0];
    const ggml_tensor * scale = mul->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(scale, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: MUL_SUM8 tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_sum8_constants constants = {
        /* .rows      = */ mul->ne[0],
        /* .n_tokens  = */ mul->ne[2],
        /* .src0_nb1  = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2  = */ static_cast<int64_t>(src0->nb[2]),
        /* .scale_nb1 = */ static_cast<int64_t>(scale->nb[1]),
        /* .scale_nb2 = */ static_cast<int64_t>(scale->nb[2]),
        /* .dst_nb1   = */ static_cast<int64_t>(dst->nb[1]),
    };

    const auto & provider = context->device_context->mul_sum8_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(((constants.rows * constants.n_tokens) + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_unary_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst,
        const ggml_backend_hrx_op_provider & provider,
        const char * op_name) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: %s tensor is not backed by a HRX buffer\n", __func__, op_name);
        return GGML_STATUS_FAILED;
    }

    const int64_t n = ggml_nelements(dst);
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &n,
            sizeof(n),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_swiglu_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: SWIGLU tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t n = ggml_nelements(dst);
    const auto & provider = context->device_context->swiglu_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &n,
            sizeof(n),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rms_norm(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: RMS_NORM tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));
    ggml_backend_hrx_rms_norm_constants constants = {
        /* .ncols   = */ src0->ne[0],
        /* .nrows   = */ ggml_nrows(src0),
        /* .ne1     = */ src0->ne[1],
        /* .ne2     = */ src0->ne[2],
        /* .src_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<int64_t>(dst->nb[3]),
        /* .eps     = */ eps,
        /* ._pad    = */ 0,
    };

    const auto & provider = context->device_context->rms_norm_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 512;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rms_norm_mul(
        ggml_backend_hrx_context * context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = rms_norm->src[0];
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(weight, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(mul, &bindings[2])) {
        GGML_LOG_ERROR("%s: RMS_NORM_MUL tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));

    ggml_backend_hrx_rms_norm_mul_constants constants = {
        /* .ncols      = */ src0->ne[0],
        /* .nrows      = */ ggml_nrows(src0),
        /* .ne1        = */ src0->ne[1],
        /* .ne2        = */ src0->ne[2],
        /* .src_nb1    = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2    = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3    = */ static_cast<int64_t>(src0->nb[3]),
        /* .weight_ne0 = */ weight->ne[0],
        /* .weight_ne1 = */ weight->ne[1],
        /* .weight_ne2 = */ weight->ne[2],
        /* .weight_ne3 = */ weight->ne[3],
        /* .weight_nb1 = */ static_cast<int64_t>(weight->nb[1]),
        /* .weight_nb2 = */ static_cast<int64_t>(weight->nb[2]),
        /* .weight_nb3 = */ static_cast<int64_t>(weight->nb[3]),
        /* .dst_nb1    = */ static_cast<int64_t>(mul->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(mul->nb[2]),
        /* .dst_nb3    = */ static_cast<int64_t>(mul->nb[3]),
        /* .eps        = */ eps,
        /* ._pad       = */ 0,
    };

    const bool use_wg128 =
        constants.ncols <= 128 &&
        context->device_context->rms_norm_mul_wg128_provider.kind == ggml_backend_hrx_provider_kind::hsaco;
    const auto & provider = use_wg128 ?
        context->device_context->rms_norm_mul_wg128_provider :
        context->device_context->rms_norm_mul_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : (use_wg128 ? 128 : 512);
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rms_norm_mul_rope_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        const ggml_tensor * rope) {
    const ggml_tensor * src0 = rms_norm->src[0];
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * pos = rope->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(weight, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(pos, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(rope, &bindings[3])) {
        GGML_LOG_ERROR("%s: RMS_NORM_MUL_ROPE tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));
    std::memcpy(&freq_base, reinterpret_cast<const int32_t *>(rope->op_params) + 5, sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const int32_t *>(rope->op_params) + 6, sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const int32_t *>(rope->op_params) + 8, sizeof(float));

    ggml_backend_hrx_rms_norm_mul_rope_constants constants = {
        /* .ncols       = */ src0->ne[0],
        /* .nrows       = */ ggml_nrows(src0),
        /* .ne1         = */ src0->ne[1],
        /* .ne2         = */ src0->ne[2],
        /* .src_nb1     = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2     = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3     = */ static_cast<int64_t>(src0->nb[3]),
        /* .weight_ne0  = */ weight->ne[0],
        /* .weight_ne1  = */ weight->ne[1],
        /* .weight_ne2  = */ weight->ne[2],
        /* .weight_ne3  = */ weight->ne[3],
        /* .weight_nb1  = */ static_cast<int64_t>(weight->nb[1]),
        /* .weight_nb2  = */ static_cast<int64_t>(weight->nb[2]),
        /* .weight_nb3  = */ static_cast<int64_t>(weight->nb[3]),
        /* .dst_nb1     = */ static_cast<int64_t>(rope->nb[1]),
        /* .dst_nb2     = */ static_cast<int64_t>(rope->nb[2]),
        /* .dst_nb3     = */ static_cast<int64_t>(rope->nb[3]),
        /* .eps         = */ eps,
        /* ._pad0       = */ 0,
        /* .n_dims      = */ ggml_get_op_params_i32(rope, 1),
        /* .mode        = */ ggml_get_op_params_i32(rope, 2),
        /* .section0    = */ ggml_get_op_params_i32(rope, 11),
        /* .section1    = */ ggml_get_op_params_i32(rope, 12),
        /* .section2    = */ ggml_get_op_params_i32(rope, 13),
        /* .section3    = */ ggml_get_op_params_i32(rope, 14),
        /* .freq_base   = */ freq_base,
        /* .freq_scale  = */ freq_scale,
        /* .attn_factor = */ attn_factor,
        /* ._pad1       = */ 0.0f,
    };

    const auto & provider = context->device_context->rms_norm_mul_rope_f32_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 512;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            4,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rms_norm_mul_rope_set_rows_f32_f16(
        ggml_backend_hrx_context * context,
        const ggml_tensor * rms_norm,
        const ggml_tensor * mul,
        const ggml_tensor * rope,
        const ggml_tensor * set_rows) {
    const ggml_tensor * src0 = rms_norm->src[0];
    const ggml_tensor * weight = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * pos = rope->src[1];
    const ggml_tensor * idxs = set_rows->src[1];
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(weight, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(pos, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(idxs, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(set_rows, &bindings[4])) {
        GGML_LOG_ERROR("%s: RMS_NORM_MUL_ROPE_SET_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&eps, rms_norm->op_params, sizeof(eps));
    std::memcpy(&freq_base, reinterpret_cast<const int32_t *>(rope->op_params) + 5, sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const int32_t *>(rope->op_params) + 6, sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const int32_t *>(rope->op_params) + 8, sizeof(float));

    ggml_backend_hrx_rms_norm_mul_rope_set_rows_constants constants = {
        /* .ncols            = */ src0->ne[0],
        /* .nrows            = */ ggml_nrows(src0),
        /* .ne1              = */ src0->ne[1],
        /* .ne2              = */ src0->ne[2],
        /* .src_nb1          = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2          = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3          = */ static_cast<int64_t>(src0->nb[3]),
        /* .weight_ne0       = */ weight->ne[0],
        /* .weight_ne1       = */ weight->ne[1],
        /* .weight_ne2       = */ weight->ne[2],
        /* .weight_ne3       = */ weight->ne[3],
        /* .weight_nb1       = */ static_cast<int64_t>(weight->nb[1]),
        /* .weight_nb2       = */ static_cast<int64_t>(weight->nb[2]),
        /* .weight_nb3       = */ static_cast<int64_t>(weight->nb[3]),
        /* .dst_nb1          = */ static_cast<int64_t>((rope->nb[1] / sizeof(float)) * sizeof(ggml_fp16_t)),
        /* .dst_nb2          = */ 0,
        /* .dst_nb3          = */ 0,
        /* .eps              = */ eps,
        /* ._pad0            = */ 0,
        /* .n_dims           = */ ggml_get_op_params_i32(rope, 1),
        /* .mode             = */ ggml_get_op_params_i32(rope, 2),
        /* .section0         = */ ggml_get_op_params_i32(rope, 11),
        /* .section1         = */ ggml_get_op_params_i32(rope, 12),
        /* .section2         = */ ggml_get_op_params_i32(rope, 13),
        /* .section3         = */ ggml_get_op_params_i32(rope, 14),
        /* .freq_base        = */ freq_base,
        /* .freq_scale       = */ freq_scale,
        /* .attn_factor      = */ attn_factor,
        /* ._pad1            = */ 0.0f,
        /* .set_rows_ne1     = */ set_rows->ne[1],
        /* .set_rows_ne11    = */ static_cast<int32_t>(idxs->ne[1]),
        /* .set_rows_ne12    = */ static_cast<int32_t>(idxs->ne[2]),
        /* .idx_nb0          = */ static_cast<int64_t>(idxs->nb[0]),
        /* .idx_nb1          = */ static_cast<int64_t>(idxs->nb[1]),
        /* .idx_nb2          = */ static_cast<int64_t>(idxs->nb[2]),
        /* .set_rows_dst_nb1 = */ static_cast<int64_t>(set_rows->nb[1]),
        /* .set_rows_dst_nb2 = */ static_cast<int64_t>(set_rows->nb[2]),
        /* .set_rows_dst_nb3 = */ static_cast<int64_t>(set_rows->nb[3]),
    };

    const auto & provider = context->device_context->rms_norm_mul_rope_set_rows_f32_f16_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 512;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_hrx_row_reduce_constants ggml_backend_hrx_row_reduce_constants_for(
        const ggml_tensor * dst,
        float eps) {
    const ggml_tensor * src0 = dst->src[0];
    return {
        /* .ncols   = */ src0->ne[0],
        /* .nrows   = */ ggml_nrows(src0),
        /* .ne1     = */ src0->ne[1],
        /* .ne2     = */ src0->ne[2],
        /* .src_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .dst_nb1 = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2 = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3 = */ static_cast<int64_t>(dst->nb[3]),
        /* .eps     = */ eps,
        /* ._pad    = */ 0,
    };
}

static ggml_status ggml_backend_hrx_dispatch_sum_rows(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: SUM_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_row_reduce_constants constants =
        ggml_backend_hrx_row_reduce_constants_for(dst, 0.0f);
    const auto & provider = context->device_context->sum_rows_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_l2_norm(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: L2_NORM tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float eps = 0.0f;
    std::memcpy(&eps, dst->op_params, sizeof(eps));
    ggml_backend_hrx_row_reduce_constants constants =
        ggml_backend_hrx_row_reduce_constants_for(dst, eps);
    const bool use_wg128 =
        constants.ncols <= 128 &&
        context->device_context->l2_norm_wg128_provider.kind == ggml_backend_hrx_provider_kind::hsaco;
    const auto & provider = use_wg128 ?
        context->device_context->l2_norm_wg128_provider :
        context->device_context->l2_norm_provider;
    if (provider.kind != ggml_backend_hrx_provider_kind::hsaco) {
        GGML_LOG_ERROR("%s: L2_NORM provider is unavailable for row length %" PRId64 "\n",
                __func__, constants.ncols);
        return GGML_STATUS_FAILED;
    }
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : (use_wg128 ? 128 : 256);
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_scale_f32(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: SCALE tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_scale_constants constants = {
        /* .n     = */ ggml_nelements(dst),
        /* .scale = */ 0.0f,
        /* .bias  = */ 0.0f,
    };
    const uint8_t * op_params = reinterpret_cast<const uint8_t *>(dst->op_params);
    std::memcpy(&constants.scale, op_params, sizeof(float));
    std::memcpy(&constants.bias, op_params + sizeof(float), sizeof(float));

    const auto & provider = context->device_context->scale_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_clamp(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: CLAMP tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_clamp_constants constants = {
        /* .n         = */ ggml_nelements(dst),
        /* .min_value = */ 0.0f,
        /* .max_value = */ 0.0f,
    };
    std::memcpy(&constants.min_value, reinterpret_cast<const float *>(dst->op_params) + 0, sizeof(float));
    std::memcpy(&constants.max_value, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));

    const auto & provider = context->device_context->clamp_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_cpy(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t src_ref = {};
    hrx_buffer_ref_t dst_ref = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &src_ref) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &dst_ref)) {
        GGML_LOG_ERROR("%s: CPY tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const size_t size = ggml_nbytes(dst);
    if (size == 0) {
        return GGML_STATUS_SUCCESS;
    }

    if (src0->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F16 &&
        ggml_is_contiguous(src0)) {
        const auto & provider = context->device_context->copy_f32_f16_provider;
        if (provider.kind != ggml_backend_hrx_provider_kind::hsaco) {
            GGML_LOG_ERROR("%s: F32->F16 CPY provider is unavailable\n", __func__);
            return GGML_STATUS_FAILED;
        }
        ggml_backend_hrx_copy_f32_f16_constants constants = {
            /* .n = */ static_cast<int64_t>(ggml_nelements(dst)),
        };
        hrx_buffer_ref_t bindings[2] = {src_ref, dst_ref};
        const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
            provider.export_info.workgroup_size[0] : 256;
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((constants.n + workgroup_size - 1) / workgroup_size),
                1,
                1,
            },
            /* .workgroup_size = */ { workgroup_size, 1, 1 },
            /* .subgroup_size = */ 0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, provider.executable, provider.export_ordinal, &config,
                &constants, sizeof(constants), bindings, 2, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
    } else if (ggml_is_contiguous(src0)) {
        if (src_ref.buffer != dst_ref.buffer || src_ref.offset != dst_ref.offset) {
            if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(
                    context->stream,
                    src_ref.buffer,
                    src_ref.offset,
                    dst_ref.buffer,
                    dst_ref.offset,
                    size))) {
                return GGML_STATUS_FAILED;
            }
        }
    } else {
        const auto & provider = context->device_context->copy_strided_f32_provider;
        if (src0->type == GGML_TYPE_F32 &&
            provider.kind == ggml_backend_hrx_provider_kind::hsaco) {
            const size_t row_size = ggml_row_size(src0->type, src0->ne[0]);
            ggml_backend_hrx_copy_strided_f32_constants constants = {
                /* .ncols    = */ src0->ne[0],
                /* .nrows    = */ ggml_nrows(src0),
                /* .ne1      = */ src0->ne[1],
                /* .ne2      = */ src0->ne[2],
                /* .src_nb1  = */ static_cast<int64_t>(src0->nb[1]),
                /* .src_nb2  = */ static_cast<int64_t>(src0->nb[2]),
                /* .src_nb3  = */ static_cast<int64_t>(src0->nb[3]),
                /* .row_size = */ static_cast<int64_t>(row_size),
            };
            hrx_buffer_ref_t bindings[2] = {src_ref, dst_ref};
            const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
                provider.export_info.workgroup_size[0] : 256;
            hrx_dispatch_config_t config = {
                /* .workgroup_count = */ {
                    static_cast<uint32_t>((constants.ncols + workgroup_size - 1) / workgroup_size),
                    static_cast<uint32_t>(constants.nrows),
                    1,
                },
                /* .workgroup_size = */ { workgroup_size, 1, 1 },
                /* .subgroup_size = */ 0,
            };
            if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                    context->stream, provider.executable, provider.export_ordinal, &config,
                    &constants, sizeof(constants), bindings, 2, HRX_DISPATCH_FLAG_NONE))) {
                return GGML_STATUS_FAILED;
            }
            return GGML_STATUS_SUCCESS;
        }

        const size_t row_size = ggml_row_size(src0->type, src0->ne[0]);
        size_t dst_offset = dst_ref.offset;
        for (int64_t i3 = 0; i3 < src0->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < src0->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < src0->ne[1]; ++i1) {
                    const size_t src_offset =
                        src_ref.offset +
                        static_cast<size_t>(i1) * src0->nb[1] +
                        static_cast<size_t>(i2) * src0->nb[2] +
                        static_cast<size_t>(i3) * src0->nb[3];
                    if (!GGML_HRX_CHECK(hrx_stream_copy_buffer(
                            context->stream,
                            src_ref.buffer,
                            src_offset,
                            dst_ref.buffer,
                            dst_offset,
                            row_size))) {
                        return GGML_STATUS_FAILED;
                    }
                    dst_offset += row_size;
                }
            }
        }
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_set_rows(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: SET_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_set_rows_constants constants = {
        /* .nc       = */ src0->ne[0],
        /* .nr       = */ src0->ne[1],
        /* .ne02     = */ src0->ne[2],
        /* .ne03     = */ src0->ne[3],
        /* .ne1      = */ dst->ne[1],
        /* .ne11     = */ src1->ne[1],
        /* .ne12     = */ src1->ne[2],
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .idx_nb0  = */ static_cast<int64_t>(src1->nb[0]),
        /* .idx_nb1  = */ static_cast<int64_t>(src1->nb[1]),
        /* .idx_nb2  = */ static_cast<int64_t>(src1->nb[2]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
    };

    const ggml_backend_hrx_op_provider * provider =
        ggml_backend_hrx_set_rows_provider(context->device_context, dst);
    if (!provider) {
        return GGML_STATUS_FAILED;
    }

    const int64_t logical_cols =
        ggml_is_quantized(dst->type) ? constants.nc / ggml_blck_size(dst->type) : constants.nc;
    const int64_t total = logical_cols * constants.nr * constants.ne02 * constants.ne03;
    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider->executable,
            provider->export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_get_rows_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    if (ggml_nelements(dst) == 0) {
        return GGML_STATUS_SUCCESS;
    }

    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: GET_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_get_rows_f32_constants constants = {
        /* .nc       = */ src0->ne[0],
        /* .nr       = */ ggml_nelements(src1),
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2 = */ static_cast<int64_t>(src0->nb[2]),
        /* .src0_nb3 = */ static_cast<int64_t>(src0->nb[3]),
        /* .idx_nb0  = */ static_cast<int64_t>(src1->nb[0]),
        /* .idx_nb1  = */ static_cast<int64_t>(src1->nb[1]),
        /* .idx_nb2  = */ static_cast<int64_t>(src1->nb[2]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
        /* .ne10     = */ src1->ne[0],
        /* .ne11     = */ src1->ne[1],
    };

    const bool use_nr1 =
        src0->type == GGML_TYPE_F32 &&
        constants.nr == 1 &&
        context->device_context->get_rows_f32_nr1_provider.kind == ggml_backend_hrx_provider_kind::hsaco;
    const auto & provider = src0->type == GGML_TYPE_Q5_K ?
        context->device_context->get_rows_q5_k_provider :
        (use_nr1 ?
            context->device_context->get_rows_f32_nr1_provider :
            context->device_context->get_rows_f32_provider);
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.nc + workgroup_size - 1) / workgroup_size),
            static_cast<uint32_t>(use_nr1 ? 1 : constants.nr),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!ggml_backend_hrx_validate_get_rows_indices(context, dst)) {
        return GGML_STATUS_FAILED;
    }

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_vec_k_q8_1(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_backend_hrx_q8_1_mmvq_variant variant =
        ggml_backend_hrx_mul_mat_vec_k_q8_1_variant(context->device_context, dst);
    if (!variant.provider) {
        GGML_LOG_ERROR("%s: K-quant x Q8_1 MUL_MAT provider is unavailable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t src1_ref = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src1, &src1_ref)) {
        GGML_LOG_ERROR("%s: Q8_1 quantize source is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t q8_1_blocks = src1->ne[1] * (src1->ne[0] / 32);
    const size_t q8_1_size = variant.x4_quant ?
        static_cast<size_t>(((q8_1_blocks + 3) / 4) * 144) :
        static_cast<size_t>(q8_1_blocks * 36);
    hrx_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_hrx_request_scratch_buffer(context, q8_1_size, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t quant_bindings[2] = { src1_ref, q8_1_ref };
    ggml_backend_hrx_quantize_q8_1_constants quant_constants = {
        /* .ne00 = */ src1->ne[0],
        /* .s01  = */ static_cast<int64_t>(src1->nb[1] / sizeof(float)),
        /* .s02  = */ static_cast<int64_t>(src1->nb[2] / sizeof(float)),
        /* .s03  = */ static_cast<int64_t>(src1->nb[3] / sizeof(float)),
        /* .ne0  = */ src1->ne[0],
        /* .ne1  = */ src1->ne[1],
        /* .ne2  = */ src1->ne[2],
    };

    const auto & quant_provider = variant.x4_quant ?
        context->device_context->quantize_q8_1_x4_provider :
        context->device_context->quantize_q8_1_provider;
    const uint32_t quant_workgroup_size = quant_provider.export_info.workgroup_size[0] ?
        quant_provider.export_info.workgroup_size[0] : (variant.x4_quant ? 128 : 32);
    hrx_dispatch_config_t quant_config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(variant.x4_quant ? quant_constants.ne0 / 128 : quant_constants.ne0 / 32),
            static_cast<uint32_t>(quant_constants.ne1),
            static_cast<uint32_t>(src1->ne[2] * src1->ne[3]),
        },
        /* .workgroup_size = */ { quant_workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };
    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            quant_provider.executable,
            quant_provider.export_ordinal,
            &quant_config,
            &quant_constants,
            sizeof(quant_constants),
            quant_bindings,
            2,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: K-quant x Q8_1 MUL_MAT tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;

    ggml_backend_hrx_mul_mat_vec_constants constants = {
        /* .k    = */ src0->ne[0],
        /* .rows = */ src0->ne[1],
        /* .cols = */ src1->ne[1],
    };

    const auto & provider = *variant.provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.rows + variant.rows_per_workgroup - 1) / variant.rows_per_workgroup),
            static_cast<uint32_t>((constants.cols + variant.cols_per_workgroup - 1) / variant.cols_per_workgroup),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            3,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_vec(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: MUL_MAT tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (ggml_backend_hrx_supports_mul_mat_vec_batched(context->device_context, dst)) {
        ggml_backend_hrx_mul_mat_vec_batched_constants constants = {
            /* .k         = */ src0->ne[0],
            /* .rows      = */ src0->ne[1],
            /* .cols      = */ src1->ne[1],
            /* .dst_ne2   = */ dst->ne[2],
            /* .dst_ne3   = */ dst->ne[3],
            /* .src0_ne2  = */ src0->ne[2],
            /* .src0_ne3  = */ src0->ne[3],
            /* .src0_nb1  = */ static_cast<int64_t>(src0->nb[1]),
            /* .src0_nb2  = */ static_cast<int64_t>(src0->nb[2]),
            /* .src0_nb3  = */ static_cast<int64_t>(src0->nb[3]),
            /* .src1_nb1  = */ static_cast<int64_t>(src1->nb[1]),
            /* .src1_nb2  = */ static_cast<int64_t>(src1->nb[2]),
            /* .src1_nb3  = */ static_cast<int64_t>(src1->nb[3]),
            /* .dst_nb1   = */ static_cast<int64_t>(dst->nb[1]),
            /* .dst_nb2   = */ static_cast<int64_t>(dst->nb[2]),
            /* .dst_nb3   = */ static_cast<int64_t>(dst->nb[3]),
        };

        const ggml_backend_hrx_op_provider * provider =
            ggml_backend_hrx_select_mul_mat_vec_batched_provider(context->device_context, dst);
        if (!provider || provider->kind != ggml_backend_hrx_provider_kind::hsaco) {
            GGML_LOG_ERROR("%s: batched MUL_MAT provider is unavailable\n", __func__);
            return GGML_STATUS_FAILED;
        }

        const uint32_t provider_rows_per_workgroup =
            provider == &context->device_context->mul_mat_vec_f32_batched_rows2_cols8_provider ? 2 : 1;
        const uint32_t provider_cols_per_workgroup =
            provider == &context->device_context->mul_mat_vec_f32_batched_rows2_cols8_provider ? 8 :
            provider == &context->device_context->mul_mat_vec_f32_batched_cols16_provider ? 16 :
            provider == &context->device_context->mul_mat_vec_f32_batched_cols8_provider ? 8 :
            provider == &context->device_context->mul_mat_vec_f16_batched_cols16_provider ? 16 :
            provider == &context->device_context->mul_mat_vec_f16_batched_cols8_provider ? 8 :
            provider == &context->device_context->mul_mat_vec_f16_batched_cols4_provider ? 4 : 1;
        const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
            provider->export_info.workgroup_size[0] : 256;
        hrx_dispatch_config_t config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>((constants.rows + provider_rows_per_workgroup - 1) / provider_rows_per_workgroup),
                static_cast<uint32_t>(
                    ((constants.cols + provider_cols_per_workgroup - 1) / provider_cols_per_workgroup) *
                    constants.dst_ne2 * constants.dst_ne3),
                1,
            },
            /* .workgroup_size = */ { workgroup_size, 1, 1 },
            /* .subgroup_size = */ 0,
        };

        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, provider->executable, provider->export_ordinal, &config,
                &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }

        return GGML_STATUS_SUCCESS;
    }

    if (ggml_backend_hrx_supports_mul_mat_vec_k_q8_1(context->device_context, dst)) {
        return ggml_backend_hrx_dispatch_mul_mat_vec_k_q8_1(context, dst);
    }

    ggml_backend_hrx_mul_mat_vec_constants constants = {
        /* .k    = */ src0->ne[0],
        /* .rows = */ src0->ne[1],
        /* .cols = */ src1->ne[1],
    };

    const ggml_backend_hrx_op_provider * provider = src0->type == GGML_TYPE_BF16 ?
        ggml_backend_hrx_select_mul_mat_vec_bf16_provider(
            context->device_context, constants.k, constants.rows, constants.cols) :
        src0->type == GGML_TYPE_Q8_0 ?
        ggml_backend_hrx_select_mul_mat_vec_q8_0_provider(context->device_context, constants.cols) :
        ggml_backend_hrx_mul_mat_vec_provider(context->device_context, src0->type);
    if (!provider || provider->kind != ggml_backend_hrx_provider_kind::hsaco) {
        GGML_LOG_ERROR("%s: MUL_MAT provider is unavailable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const uint32_t provider_cols_per_workgroup =
        provider == &context->device_context->mul_mat_vec_bf16_wmma16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_cols32_provider ? 32 :
        provider == &context->device_context->mul_mat_vec_bf16_rows2_cols16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_cols16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_cols8_provider ? 8 :
        provider == &context->device_context->mul_mat_vec_bf16_cols4_provider ? 4 :
        provider == &context->device_context->mul_mat_vec_q8_0_cols8_provider ? 8 : 1;
    const uint32_t provider_rows_per_workgroup =
        provider == &context->device_context->mul_mat_vec_bf16_wmma16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_rows4_k512_cols1_provider ? 4 :
        provider == &context->device_context->mul_mat_vec_bf16_rows4_k2048_cols1_provider ? 4 :
        provider == &context->device_context->mul_mat_vec_bf16_rows2_cols1_wg32_provider ? 2 :
        provider == &context->device_context->mul_mat_vec_bf16_rows2_cols1_provider ? 2 :
        provider == &context->device_context->mul_mat_vec_bf16_rows2_cols16_provider ? 2 : 1;
    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.rows + provider_rows_per_workgroup - 1) / provider_rows_per_workgroup),
            static_cast<uint32_t>((constants.cols + provider_cols_per_workgroup - 1) / provider_cols_per_workgroup),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider->executable, provider->export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_vec_q8_0_add(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mm,
        const ggml_tensor * add) {
    const ggml_tensor * bias = add->src[0] == mm ? add->src[1] : add->src[0];
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(bias, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(add, &bindings[3])) {
        GGML_LOG_ERROR("%s: fused MUL_MAT_ADD tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_mat_vec_constants constants = {
        /* .k    = */ src0->ne[0],
        /* .rows = */ src0->ne[1],
        /* .cols = */ src1->ne[1],
    };

    const bool use_q8_1_x4_mmq128x32 =
        ggml_backend_hrx_supports_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_prompt(
            context->device_context,
            mm,
            context->device_context->mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider,
            "GGML_HRX_DISABLE_Q8_0_ADD_Q8_1_X4_MMQ128X32_PROMPT");
    const bool use_rows4_cols4 =
        !use_q8_1_x4_mmq128x32 &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_0_ADD_ROWS4_COLS4_PROMPT") &&
        constants.cols == 512 &&
        (constants.rows % 4) == 0 &&
        ggml_backend_hrx_provider_available(context->device_context->mul_mat_vec_q8_0_add_rows4_cols4_provider);
    const bool use_cols8 =
        !use_q8_1_x4_mmq128x32 &&
        !use_rows4_cols4 &&
        !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_Q8_0_COLS8_PROMPT") &&
        constants.cols == 512 &&
        ggml_backend_hrx_provider_available(context->device_context->mul_mat_vec_q8_0_add_cols8_provider);

    const auto & provider = use_q8_1_x4_mmq128x32 ?
        context->device_context->mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_provider :
        use_rows4_cols4 ?
        context->device_context->mul_mat_vec_q8_0_add_rows4_cols4_provider :
        use_cols8 ?
        context->device_context->mul_mat_vec_q8_0_add_cols8_provider :
        context->device_context->mul_mat_vec_q8_0_add_provider;
    if (!ggml_backend_hrx_provider_available(provider)) {
        GGML_LOG_ERROR("%s: fused MUL_MAT_ADD provider is unavailable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (use_q8_1_x4_mmq128x32) {
        const int64_t q8_1_blocks = src1->ne[1] * (src1->ne[0] / 32);
        const size_t q8_1_size = static_cast<size_t>(((q8_1_blocks + 3) / 4) * 144);
        hrx_buffer_ref_t q8_1_ref = {};
        if (!ggml_backend_hrx_request_scratch_buffer(context, q8_1_size, &q8_1_ref)) {
            return GGML_STATUS_FAILED;
        }

        hrx_buffer_ref_t quant_bindings[2] = { bindings[1], q8_1_ref };
        ggml_backend_hrx_quantize_q8_1_constants quant_constants = {
            /* .ne00 = */ src1->ne[0],
            /* .s01  = */ static_cast<int64_t>(src1->nb[1] / sizeof(float)),
            /* .s02  = */ static_cast<int64_t>(src1->nb[2] / sizeof(float)),
            /* .s03  = */ static_cast<int64_t>(src1->nb[3] / sizeof(float)),
            /* .ne0  = */ src1->ne[0],
            /* .ne1  = */ src1->ne[1],
            /* .ne2  = */ src1->ne[2],
        };
        const auto & quant_provider = context->device_context->quantize_q8_1_x4_provider;
        const uint32_t quant_workgroup_size = quant_provider.export_info.workgroup_size[0] ?
            quant_provider.export_info.workgroup_size[0] : 128;
        hrx_dispatch_config_t quant_config = {
            /* .workgroup_count = */ {
                static_cast<uint32_t>(quant_constants.ne0 / 128),
                static_cast<uint32_t>(quant_constants.ne1),
                static_cast<uint32_t>(src1->ne[2] * src1->ne[3]),
            },
            /* .workgroup_size = */ { quant_workgroup_size, 1, 1 },
            /* .subgroup_size = */ 0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream,
                quant_provider.executable,
                quant_provider.export_ordinal,
                &quant_config,
                &quant_constants,
                sizeof(quant_constants),
                quant_bindings,
                2,
                HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        bindings[1] = q8_1_ref;
    }

    const uint32_t provider_rows_per_workgroup = use_q8_1_x4_mmq128x32 ? 128 : (use_rows4_cols4 ? 4 : 1);
    const uint32_t provider_cols_per_workgroup = use_q8_1_x4_mmq128x32 ? 32 : (use_rows4_cols4 ? 4 : (use_cols8 ? 8 : 1));
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.rows + provider_rows_per_workgroup - 1) / provider_rows_per_workgroup),
            static_cast<uint32_t>((constants.cols + provider_cols_per_workgroup - 1) / provider_cols_per_workgroup),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider.executable, provider.export_ordinal, &config,
        &constants, sizeof(constants), bindings, 4, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_vec_bf16_swiglu(
        ggml_backend_hrx_context * context,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * swiglu) {
    const ggml_tensor * src1 = up->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(gate->src[0], &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(up->src[0], &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(swiglu, &bindings[3])) {
        GGML_LOG_ERROR("%s: MUL_MAT_SWIGLU tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_mat_vec_constants constants = {
        /* .k    = */ up->src[0]->ne[0],
        /* .rows = */ up->src[0]->ne[1],
        /* .cols = */ src1->ne[1],
    };

    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_vec_bf16_swiglu_provider(
        context->device_context, constants.k, constants.rows, constants.cols);
    if (!provider || provider->kind != ggml_backend_hrx_provider_kind::hsaco) {
        GGML_LOG_ERROR("%s: MUL_MAT_SWIGLU provider is unavailable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const uint32_t provider_rows_per_workgroup =
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_wmma16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_provider ? 4 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_rows2_cols1_provider ? 2 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider ? 2 : 1;
    const uint32_t provider_cols_per_workgroup =
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_wmma16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_rows2_cols8_provider ? 8 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_cols16_provider ? 16 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_cols8_provider ? 8 :
        provider == &context->device_context->mul_mat_vec_bf16_swiglu_cols4_provider ? 4 : 1;
    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.rows + provider_rows_per_workgroup - 1) / provider_rows_per_workgroup),
            static_cast<uint32_t>((constants.cols + provider_cols_per_workgroup - 1) / provider_cols_per_workgroup),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider->executable, provider->export_ordinal, &config,
        &constants, sizeof(constants), bindings, 4, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_vec_bf16_set_rows_f16(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mul_mat,
        const ggml_tensor * set_rows) {
    const ggml_tensor * src0 = mul_mat->src[0];
    const ggml_tensor * src1 = mul_mat->src[1];
    const ggml_tensor * idxs = set_rows->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(idxs, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(set_rows, &bindings[3])) {
        GGML_LOG_ERROR("%s: MUL_MAT_SET_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_mat_vec_bf16_set_rows_constants constants = {
        /* .k            = */ src0->ne[0],
        /* .rows         = */ src0->ne[1],
        /* .set_rows_ne1 = */ set_rows->ne[1],
        /* .idx_nb0      = */ static_cast<int64_t>(idxs->nb[0]),
        /* .dst_nb1      = */ static_cast<int64_t>(set_rows->nb[1]),
    };

    const auto & provider = context->device_context->mul_mat_vec_bf16_set_rows_f16_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.rows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider.executable, provider.export_ordinal, &config,
        &constants, sizeof(constants), bindings, 4, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_id_q4_k(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[3])) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_mul_mat_id_q4_k_constants constants = {
        /* .k          = */ src0->ne[0],
        /* .rows       = */ src0->ne[1],
        /* .n_ids      = */ src2->ne[0],
        /* .n_tokens   = */ src2->ne[1],
        /* .n_experts  = */ src0->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .src1_nb1   = */ src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]),
        /* .src1_nb2   = */ static_cast<int64_t>(src1->nb[2]),
        /* .ids_nb0    = */ static_cast<int64_t>(src2->nb[0]),
        /* .ids_nb1    = */ static_cast<int64_t>(src2->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(dst->nb[2]),
    };

    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_id_q4_k_provider(
        context->device_context, constants.k, constants.rows, constants.n_ids, constants.n_tokens);
    if (!provider || provider->kind != ggml_backend_hrx_provider_kind::hsaco) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID provider is unavailable\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const bool use_grouped =
        provider == &context->device_context->mul_mat_id_q4_k_grouped_row2_route8_wg64_provider ||
        provider == &context->device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider;
    const bool use_q8_1_x4_mmq =
        provider == &context->device_context->mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_provider;
    if (use_grouped) {
        const size_t route_capacity = static_cast<size_t>(constants.n_ids * constants.n_tokens);
        const size_t counts_size = static_cast<size_t>(constants.n_experts) * sizeof(uint32_t);
        const size_t routes_size = static_cast<size_t>(constants.n_experts) * route_capacity * sizeof(uint32_t);
        hrx_buffer_ref_t scratch_ref = {};
        if (!ggml_backend_hrx_request_scratch_buffer(context, counts_size + routes_size, &scratch_ref)) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_ref_t counts_ref = { scratch_ref.buffer, scratch_ref.offset, counts_size };
        hrx_buffer_ref_t routes_ref = { scratch_ref.buffer, scratch_ref.offset + counts_size, routes_size };

        const auto & clear_provider = context->device_context->clear_u32_provider;
        ggml_backend_hrx_clear_u32_constants clear_constants = { constants.n_experts };
        hrx_dispatch_config_t clear_config = {
            { static_cast<uint32_t>((clear_constants.n + 255) / 256), 1, 1 },
            { clear_provider.export_info.workgroup_size[0] ? clear_provider.export_info.workgroup_size[0] : 256, 1, 1 },
            0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, clear_provider.executable, clear_provider.export_ordinal, &clear_config,
                &clear_constants, sizeof(clear_constants), &counts_ref, 1, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }

        hrx_buffer_ref_t compact_bindings[3] = { bindings[2], counts_ref, routes_ref };
        ggml_backend_hrx_compact_moe_routes_constants compact_constants = {
            constants.n_ids,
            constants.n_tokens,
            constants.n_experts,
            static_cast<int64_t>(route_capacity),
            constants.ids_nb0,
            constants.ids_nb1,
        };
        const auto & compact_provider = context->device_context->compact_moe_routes_provider;
        hrx_dispatch_config_t compact_config = {
            { static_cast<uint32_t>((compact_constants.n_ids * compact_constants.n_tokens + 255) / 256), 1, 1 },
            { compact_provider.export_info.workgroup_size[0] ? compact_provider.export_info.workgroup_size[0] : 256, 1, 1 },
            0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, compact_provider.executable, compact_provider.export_ordinal, &compact_config,
                &compact_constants, sizeof(compact_constants), compact_bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }

        hrx_buffer_ref_t src1_compute_ref = bindings[1];
        if (use_q8_1_x4_mmq) {
            const int64_t q8_1_blocks = constants.n_tokens * constants.n_ids * (constants.k / 32);
            const size_t q8_1_size = static_cast<size_t>(((q8_1_blocks + 3) / 4) * 144);
            if (!ggml_backend_hrx_request_scratch_buffer(context, q8_1_size, &src1_compute_ref)) {
                return GGML_STATUS_FAILED;
            }
            hrx_buffer_ref_t quant_bindings[2] = { bindings[1], src1_compute_ref };
            ggml_backend_hrx_quantize_q8_1_constants quant_constants = {
                src1->ne[0],
                src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1] / sizeof(float)),
                static_cast<int64_t>(src1->nb[2] / sizeof(float)),
                static_cast<int64_t>(src1->nb[3] / sizeof(float)),
                constants.k,
                constants.n_ids,
                constants.n_tokens,
            };
            const auto & quant_provider = context->device_context->quantize_q8_1_x4_provider;
            hrx_dispatch_config_t quant_config = {
                { static_cast<uint32_t>((quant_constants.ne0 + 127) / 128),
                  static_cast<uint32_t>(quant_constants.ne1),
                  static_cast<uint32_t>(quant_constants.ne2) },
                { quant_provider.export_info.workgroup_size[0] ? quant_provider.export_info.workgroup_size[0] : 128, 1, 1 },
                0,
            };
            if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                    context->stream, quant_provider.executable, quant_provider.export_ordinal, &quant_config,
                    &quant_constants, sizeof(quant_constants), quant_bindings, 2, HRX_DISPATCH_FLAG_NONE))) {
                return GGML_STATUS_FAILED;
            }
        }

        hrx_buffer_ref_t grouped_bindings[5] = { bindings[0], src1_compute_ref, counts_ref, routes_ref, bindings[3] };
        ggml_backend_hrx_mul_mat_id_q4_k_grouped_constants grouped_constants = {
            constants.k,
            constants.rows,
            constants.n_ids,
            constants.n_tokens,
            constants.n_experts,
            static_cast<int64_t>(route_capacity),
            constants.src0_nb1,
            constants.src0_nb2,
            constants.src1_nb1,
            constants.src1_nb2,
            constants.dst_nb1,
            constants.dst_nb2,
        };
        hrx_dispatch_config_t grouped_config = {
            { static_cast<uint32_t>(use_q8_1_x4_mmq ? (grouped_constants.rows + 63) / 64 : (grouped_constants.rows + 1) / 2),
              static_cast<uint32_t>(use_q8_1_x4_mmq ? (grouped_constants.n_tokens + 31) / 32 : grouped_constants.n_experts),
              static_cast<uint32_t>(use_q8_1_x4_mmq ? grouped_constants.n_experts : 1) },
            { provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : 64, 1, 1 },
            0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, provider->executable, provider->export_ordinal, &grouped_config,
                &grouped_constants, sizeof(grouped_constants), grouped_bindings, 5, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        return GGML_STATUS_SUCCESS;
    }

    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(
                provider == &context->device_context->mul_mat_id_q4_k_row4_wg64_provider ?
                    (constants.rows + 3) / 4 : constants.rows),
            static_cast<uint32_t>(constants.n_ids * constants.n_tokens),
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider->executable, provider->export_ordinal, &config,
            &constants, sizeof(constants), bindings, 4, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static bool ggml_backend_hrx_dispatch_quantize_q8_1(
        ggml_backend_hrx_context * context,
        const ggml_tensor * src,
        hrx_buffer_ref_t * q8_1_ref) {
    hrx_buffer_ref_t src_ref = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src, &src_ref)) {
        return false;
    }
    const int64_t q8_1_blocks = src->ne[1] * src->ne[2] * (src->ne[0] / 32);
    const size_t q8_1_size = static_cast<size_t>(q8_1_blocks * 36);
    if (!ggml_backend_hrx_request_scratch_buffer(context, q8_1_size, q8_1_ref)) {
        return false;
    }
    ggml_backend_hrx_quantize_q8_1_constants constants = {
        src->ne[0],
        static_cast<int64_t>(src->nb[1] / sizeof(float)),
        static_cast<int64_t>(src->nb[2] / sizeof(float)),
        static_cast<int64_t>(src->nb[3] / sizeof(float)),
        src->ne[0],
        src->ne[1],
        src->ne[2],
    };
    const auto & provider = context->device_context->quantize_q8_1_provider;
    hrx_buffer_ref_t bindings[2] = { src_ref, *q8_1_ref };
    hrx_dispatch_config_t config = {
        { static_cast<uint32_t>(constants.ne0 / 32),
          static_cast<uint32_t>(constants.ne1),
          static_cast<uint32_t>(constants.ne2) },
        { provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 32, 1, 1 },
        0,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider.executable, provider.export_ordinal, &config,
        &constants, sizeof(constants), bindings, 2, HRX_DISPATCH_FLAG_NONE));
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_id_q4_k_q8_1(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    hrx_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_hrx_dispatch_quantize_q8_1(context, src1, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[3])) {
        GGML_LOG_ERROR("%s: Q8_1 MUL_MAT_ID tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;
    ggml_backend_hrx_mul_mat_id_q4_k_q8_1_constants constants = {
        src0->ne[0], src0->ne[1], src2->ne[0], src2->ne[1], src0->ne[2],
        static_cast<int64_t>(src0->nb[1]), static_cast<int64_t>(src0->nb[2]),
        static_cast<int64_t>(src2->nb[0]), static_cast<int64_t>(src2->nb[1]),
        static_cast<int64_t>(dst->nb[1]), static_cast<int64_t>(dst->nb[2]), src1->ne[1],
    };
    const auto & provider = context->device_context->mul_mat_id_q4_k_q8_1_provider;
    hrx_dispatch_config_t config = {
        { static_cast<uint32_t>(constants.rows), static_cast<uint32_t>(constants.n_ids * constants.n_tokens), 1 },
        { provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256, 1, 1 },
        0,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider.executable, provider.export_ordinal, &config,
        &constants, sizeof(constants), bindings, 4, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_id_q4_k_mul(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * src2 = mmid->src[2];
    const ggml_tensor * scale = mul->src[1];
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(scale, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID_MUL tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    ggml_backend_hrx_mul_mat_id_q4_k_mul_constants constants = {
        src0->ne[0], src0->ne[1], src2->ne[0], src2->ne[1], src0->ne[2],
        static_cast<int64_t>(src0->nb[1]), static_cast<int64_t>(src0->nb[2]),
        src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]), static_cast<int64_t>(src1->nb[2]),
        static_cast<int64_t>(src2->nb[0]), static_cast<int64_t>(src2->nb[1]),
        static_cast<int64_t>(mul->nb[1]), static_cast<int64_t>(mul->nb[2]), static_cast<int64_t>(scale->nb[1]),
    };
    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_id_q4_k_mul_provider(
        context->device_context, constants.k, constants.rows, constants.n_ids, constants.n_tokens);
    const bool rows2 =
        provider == &context->device_context->mul_mat_id_q4_k_mul_rows2_x16_wg32_provider;
    hrx_dispatch_config_t config = {
        { static_cast<uint32_t>(rows2 ? (constants.rows + 1) / 2 : constants.rows),
          static_cast<uint32_t>(constants.n_ids * constants.n_tokens), 1 },
        { provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : 256, 1, 1 },
        0,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider->executable, provider->export_ordinal, &config,
        &constants, sizeof(constants), bindings, 5, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_id_q4_k_mul_q8_1(
        ggml_backend_hrx_context * context,
        const ggml_tensor * mmid,
        const ggml_tensor * mul) {
    const ggml_tensor * src0 = mmid->src[0];
    const ggml_tensor * src1 = mmid->src[1];
    const ggml_tensor * src2 = mmid->src[2];
    const ggml_tensor * scale = mul->src[1];
    hrx_buffer_ref_t q8_1_ref = {};
    if (!ggml_backend_hrx_dispatch_quantize_q8_1(context, src1, &q8_1_ref)) {
        return GGML_STATUS_FAILED;
    }
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src2, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(scale, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(mul, &bindings[4])) {
        GGML_LOG_ERROR("%s: Q8_1 MUL_MAT_ID_MUL tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    bindings[1] = q8_1_ref;
    ggml_backend_hrx_mul_mat_id_q4_k_mul_q8_1_constants constants = {
        src0->ne[0], src0->ne[1], src2->ne[0], src2->ne[1], src0->ne[2],
        static_cast<int64_t>(src0->nb[1]), static_cast<int64_t>(src0->nb[2]),
        static_cast<int64_t>(src2->nb[0]), static_cast<int64_t>(src2->nb[1]),
        static_cast<int64_t>(mul->nb[1]), static_cast<int64_t>(mul->nb[2]), src1->ne[1],
        static_cast<int64_t>(scale->nb[1]),
    };
    const auto & provider = context->device_context->mul_mat_id_q4_k_mul_q8_1_provider;
    hrx_dispatch_config_t config = {
        { static_cast<uint32_t>(constants.rows), static_cast<uint32_t>(constants.n_ids * constants.n_tokens), 1 },
        { provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256, 1, 1 },
        0,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider.executable, provider.export_ordinal, &config,
        &constants, sizeof(constants), bindings, 5, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_mul_mat_id_q4_k_swiglu(
        ggml_backend_hrx_context * context,
        const ggml_tensor * gate,
        const ggml_tensor * up,
        const ggml_tensor * swiglu) {
    const ggml_tensor * src1 = up->src[1];
    const ggml_tensor * ids = up->src[2];
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(gate->src[0], &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(up->src[0], &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(ids, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(swiglu, &bindings[4])) {
        GGML_LOG_ERROR("%s: MUL_MAT_ID_SWIGLU tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    ggml_backend_hrx_mul_mat_id_q4_k_swiglu_constants constants = {
        up->src[0]->ne[0], up->src[0]->ne[1], ids->ne[0], ids->ne[1], up->src[0]->ne[2],
        static_cast<int64_t>(gate->src[0]->nb[1]), static_cast<int64_t>(gate->src[0]->nb[2]),
        static_cast<int64_t>(up->src[0]->nb[1]), static_cast<int64_t>(up->src[0]->nb[2]),
        src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1]), static_cast<int64_t>(src1->nb[2]),
        static_cast<int64_t>(ids->nb[0]), static_cast<int64_t>(ids->nb[1]),
        static_cast<int64_t>(swiglu->nb[1]), static_cast<int64_t>(swiglu->nb[2]),
    };
    const ggml_backend_hrx_op_provider * provider = ggml_backend_hrx_select_mul_mat_id_q4_k_swiglu_provider(
        context->device_context, constants.k, constants.rows, constants.n_ids, constants.n_tokens);
    const bool use_grouped =
        provider == &context->device_context->mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_provider ||
        provider == &context->device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider;
    const bool use_q8_1_x4_mmq =
        provider == &context->device_context->mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_provider;
    if (use_grouped) {
        const size_t route_capacity = static_cast<size_t>(constants.n_ids * constants.n_tokens);
        const size_t counts_size = static_cast<size_t>(constants.n_experts) * sizeof(uint32_t);
        const size_t routes_size = static_cast<size_t>(constants.n_experts) * route_capacity * sizeof(uint32_t);
        hrx_buffer_ref_t scratch_ref = {};
        if (!ggml_backend_hrx_request_scratch_buffer(context, counts_size + routes_size, &scratch_ref)) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_ref_t counts_ref = { scratch_ref.buffer, scratch_ref.offset, counts_size };
        hrx_buffer_ref_t routes_ref = { scratch_ref.buffer, scratch_ref.offset + counts_size, routes_size };
        const auto & clear_provider = context->device_context->clear_u32_provider;
        ggml_backend_hrx_clear_u32_constants clear_constants = { constants.n_experts };
        hrx_dispatch_config_t clear_config = {
            { static_cast<uint32_t>((clear_constants.n + 255) / 256), 1, 1 },
            { clear_provider.export_info.workgroup_size[0] ? clear_provider.export_info.workgroup_size[0] : 256, 1, 1 },
            0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, clear_provider.executable, clear_provider.export_ordinal, &clear_config,
                &clear_constants, sizeof(clear_constants), &counts_ref, 1, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_ref_t compact_bindings[3] = { bindings[3], counts_ref, routes_ref };
        ggml_backend_hrx_compact_moe_routes_constants compact_constants = {
            constants.n_ids, constants.n_tokens, constants.n_experts, static_cast<int64_t>(route_capacity),
            constants.ids_nb0, constants.ids_nb1,
        };
        const auto & compact_provider = context->device_context->compact_moe_routes_provider;
        hrx_dispatch_config_t compact_config = {
            { static_cast<uint32_t>((compact_constants.n_ids * compact_constants.n_tokens + 255) / 256), 1, 1 },
            { compact_provider.export_info.workgroup_size[0] ? compact_provider.export_info.workgroup_size[0] : 256, 1, 1 },
            0,
        };
        if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                context->stream, compact_provider.executable, compact_provider.export_ordinal, &compact_config,
                &compact_constants, sizeof(compact_constants), compact_bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
            return GGML_STATUS_FAILED;
        }
        hrx_buffer_ref_t src1_compute_ref = bindings[2];
        if (use_q8_1_x4_mmq) {
            const int64_t q8_1_blocks = constants.n_tokens * constants.n_ids * (constants.k / 32);
            const size_t q8_1_size = static_cast<size_t>(((q8_1_blocks + 3) / 4) * 144);
            if (!ggml_backend_hrx_request_scratch_buffer(context, q8_1_size, &src1_compute_ref)) {
                return GGML_STATUS_FAILED;
            }
            hrx_buffer_ref_t quant_bindings[2] = { bindings[2], src1_compute_ref };
            ggml_backend_hrx_quantize_q8_1_constants quant_constants = {
                src1->ne[0],
                src1->ne[1] == 1 ? 0 : static_cast<int64_t>(src1->nb[1] / sizeof(float)),
                static_cast<int64_t>(src1->nb[2] / sizeof(float)),
                static_cast<int64_t>(src1->nb[3] / sizeof(float)),
                constants.k,
                constants.n_ids,
                constants.n_tokens,
            };
            const auto & quant_provider = context->device_context->quantize_q8_1_x4_provider;
            hrx_dispatch_config_t quant_config = {
                { static_cast<uint32_t>((quant_constants.ne0 + 127) / 128),
                  static_cast<uint32_t>(quant_constants.ne1),
                  static_cast<uint32_t>(quant_constants.ne2) },
                { quant_provider.export_info.workgroup_size[0] ? quant_provider.export_info.workgroup_size[0] : 128, 1, 1 },
                0,
            };
            if (!GGML_HRX_CHECK(hrx_stream_dispatch(
                    context->stream, quant_provider.executable, quant_provider.export_ordinal, &quant_config,
                    &quant_constants, sizeof(quant_constants), quant_bindings, 2, HRX_DISPATCH_FLAG_NONE))) {
                return GGML_STATUS_FAILED;
            }
        }
        hrx_buffer_ref_t grouped_bindings[6] = {
            bindings[0], bindings[1], src1_compute_ref, counts_ref, routes_ref, bindings[4] };
        ggml_backend_hrx_mul_mat_id_q4_k_swiglu_grouped_constants grouped_constants = {
            constants.k, constants.rows, constants.n_ids, constants.n_tokens, constants.n_experts,
            static_cast<int64_t>(route_capacity), constants.gate_nb1, constants.gate_nb2, constants.up_nb1,
            constants.up_nb2, constants.src1_nb1, constants.src1_nb2, constants.dst_nb1, constants.dst_nb2,
        };
        hrx_dispatch_config_t grouped_config = {
            { static_cast<uint32_t>(use_q8_1_x4_mmq ? (grouped_constants.rows + 15) / 16 : (grouped_constants.rows + 1) / 2),
              static_cast<uint32_t>(use_q8_1_x4_mmq ? (grouped_constants.n_tokens + 31) / 32 : grouped_constants.n_experts),
              static_cast<uint32_t>(use_q8_1_x4_mmq ? grouped_constants.n_experts : 1) },
            { provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : 64, 1, 1 },
            0,
        };
        return GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider->executable, provider->export_ordinal, &grouped_config,
            &grouped_constants, sizeof(grouped_constants), grouped_bindings, 6, HRX_DISPATCH_FLAG_NONE)) ?
            GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
    }
    const bool row4 = provider == &context->device_context->mul_mat_id_q4_k_swiglu_row4_wg64_provider;
    hrx_dispatch_config_t config = {
        { static_cast<uint32_t>(row4 ? (constants.rows + 3) / 4 : constants.rows),
          static_cast<uint32_t>(constants.n_ids * constants.n_tokens), 1 },
        { provider->export_info.workgroup_size[0] ? provider->export_info.workgroup_size[0] : 256, 1, 1 },
        0,
    };
    return GGML_HRX_CHECK(hrx_stream_dispatch(
        context->stream, provider->executable, provider->export_ordinal, &config,
        &constants, sizeof(constants), bindings, 5, HRX_DISPATCH_FLAG_NONE)) ?
        GGML_STATUS_SUCCESS : GGML_STATUS_FAILED;
}

static ggml_status ggml_backend_hrx_dispatch_flash_attn_ext_f32_decode(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    hrx_buffer_ref_t bindings[6] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(q, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(k, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(v, &bindings[2]) ||
        (mask && !ggml_backend_hrx_tensor_buffer_ref(mask, &bindings[3])) ||
        (sinks && !ggml_backend_hrx_tensor_buffer_ref(sinks, &bindings[4])) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[5])) {
        GGML_LOG_ERROR("%s: FLASH_ATTN_EXT tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (!mask) {
        bindings[3] = bindings[0];
    }
    if (!sinks) {
        bindings[4] = bindings[0];
    }

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    std::memcpy(&scale, reinterpret_cast<const int32_t *>(dst->op_params), sizeof(float));
    std::memcpy(&max_bias, reinterpret_cast<const int32_t *>(dst->op_params) + 1, sizeof(float));
    std::memcpy(&logit_softcap, reinterpret_cast<const int32_t *>(dst->op_params) + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const int32_t n_head_log2 =
        1 << static_cast<int32_t>(std::floor(std::log2(static_cast<double>(q->ne[2]))));
    const float m0 = std::pow(2.0f, -max_bias / n_head_log2);
    const float m1 = std::pow(2.0f, -(max_bias / 2.0f) / n_head_log2);
    ggml_backend_hrx_flash_attn_ext_f32_decode_constants constants = {
        /* .D        = */ q->ne[0],
        /* .KV       = */ k->ne[1],
        /* .N        = */ q->ne[1],
        /* .H        = */ q->ne[2],
        /* .H_KV     = */ k->ne[2],
        /* .S        = */ q->ne[3],
        /* .q_nb1    = */ static_cast<int64_t>(q->nb[1]),
        /* .q_nb2    = */ static_cast<int64_t>(q->nb[2]),
        /* .q_nb3    = */ static_cast<int64_t>(q->nb[3]),
        /* .k_nb1    = */ static_cast<int64_t>(k->nb[1]),
        /* .k_nb2    = */ static_cast<int64_t>(k->nb[2]),
        /* .k_nb3    = */ static_cast<int64_t>(k->nb[3]),
        /* .v_nb1    = */ static_cast<int64_t>(v->nb[1]),
        /* .v_nb2    = */ static_cast<int64_t>(v->nb[2]),
        /* .v_nb3    = */ static_cast<int64_t>(v->nb[3]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2  = */ static_cast<int64_t>(dst->nb[2]),
        /* .dst_nb3  = */ static_cast<int64_t>(dst->nb[3]),
        /* .mask_nb0 = */ mask ? static_cast<int64_t>(mask->nb[0]) : 0,
        /* .mask_nb1 = */ mask ? static_cast<int64_t>(mask->nb[1]) : 0,
        /* .mask_nb3 = */ mask ? static_cast<int64_t>(mask->nb[3]) : 0,
        /* .scale    = */ scale,
        /* .has_mask = */ mask ? 1 : 0,
        /* .max_bias = */ max_bias,
        /* .m0       = */ m0,
        /* .m1       = */ m1,
        /* .logit_softcap = */ logit_softcap,
        /* .n_head_log2 = */ n_head_log2,
        /* .has_sinks = */ sinks ? 1 : 0,
    };

    const bool use_prefill_direct =
        ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_direct(context->device_context, dst);
    const bool use_prefill_wmma = !use_prefill_direct &&
        ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_wmma(context->device_context, dst);
    const bool use_prefill_tile = !use_prefill_direct && !use_prefill_wmma &&
        ggml_backend_hrx_supports_flash_attn_ext_f32_f16_prefill_tile(context->device_context, dst);
    const ggml_backend_hrx_op_provider * provider =
        use_prefill_direct ? &context->device_context->flash_attn_ext_f16_prefill_direct_provider :
        use_prefill_wmma ? &context->device_context->flash_attn_ext_f16_prefill_wmma_provider :
        use_prefill_tile ? &context->device_context->flash_attn_ext_f16_prefill_tile_provider :
        ggml_backend_hrx_flash_attn_ext_f32_decode_provider(context->device_context, k, v);
    if (!provider || provider->kind != ggml_backend_hrx_provider_kind::hsaco) {
        GGML_LOG_ERROR("%s: FLASH_ATTN_EXT K/V type is unsupported\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const uint32_t workgroup_size = provider->export_info.workgroup_size[0] ?
        provider->export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>(
                (use_prefill_direct || use_prefill_wmma) ? ((constants.N + 15) / 16) :
                use_prefill_tile ? ((constants.N + 7) / 8) : constants.H),
            static_cast<uint32_t>((use_prefill_direct || use_prefill_wmma || use_prefill_tile) ?
                constants.H : constants.N),
            static_cast<uint32_t>(constants.S),
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider->executable, provider->export_ordinal, &config,
            &constants, sizeof(constants), bindings, 6, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_concat_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: CONCAT tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_concat_f32_constants constants = {
        /* .ne0      = */ dst->ne[0],
        /* .ne1      = */ dst->ne[1],
        /* .src0_ne0 = */ src0->ne[0],
        /* .src0_nb0 = */ static_cast<int64_t>(src0->nb[0]),
        /* .src0_nb1 = */ static_cast<int64_t>(src0->nb[1]),
        /* .src1_nb0 = */ static_cast<int64_t>(src1->nb[0]),
        /* .src1_nb1 = */ static_cast<int64_t>(src1->nb[1]),
        /* .dst_nb0  = */ static_cast<int64_t>(dst->nb[0]),
        /* .dst_nb1  = */ static_cast<int64_t>(dst->nb[1]),
    };

    const auto & provider = context->device_context->concat_f32_provider;
    const int64_t n = constants.ne0 * constants.ne1;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((n + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_soft_max_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, src1 ? &bindings[2] : &bindings[1])) {
        GGML_LOG_ERROR("%s: SOFT_MAX tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }
    if (src1 && !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1])) {
        GGML_LOG_ERROR("%s: SOFT_MAX mask tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const int32_t *>(dst->op_params), sizeof(float));
    ggml_backend_hrx_soft_max_f32_constants constants = {
        /* .ncols    = */ src0->ne[0],
        /* .nrows    = */ ggml_nrows(src0),
        /* .ne01     = */ src0->ne[1],
        /* .ne02     = */ src0->ne[2],
        /* .mask_nb1 = */ src1 ? static_cast<int64_t>(src1->nb[1]) : 0,
        /* .mask_nb2 = */ src1 ? static_cast<int64_t>(src1->nb[2]) : 0,
        /* .mask_nb3 = */ src1 ? static_cast<int64_t>(src1->nb[3]) : 0,
        /* .mask_ne1 = */ src1 ? src1->ne[1] : 1,
        /* .mask_ne2 = */ src1 ? src1->ne[2] : 1,
        /* .mask_ne3 = */ src1 ? src1->ne[3] : 1,
        /* .scale    = */ scale,
        /* ._pad     = */ 0,
    };

    const auto & provider = src1 ?
        context->device_context->soft_max_f32_mask_provider :
        context->device_context->soft_max_f32_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, src1 ? 3 : 2, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_argsort_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    hrx_buffer_ref_t bindings[2] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[1])) {
        GGML_LOG_ERROR("%s: ARGSORT tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_argsort_f32_constants constants = {
        /* .ncols     = */ src0->ne[0],
        /* .nrows     = */ ggml_nrows(src0),
        /* .order     = */ ggml_get_op_params_i32(dst, 0),
        /* .ncols_pad = */ ggml_backend_hrx_next_power_of_2(src0->ne[0]),
    };

    const auto & provider = context->device_context->argsort_f32_provider;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ { static_cast<uint32_t>(constants.nrows), 1, 1 },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 256,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 2, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_topk_moe_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * soft_max,
        const ggml_tensor * weights,
        const ggml_tensor * ids,
        const ggml_tensor * clamp) {
    const ggml_tensor * logits = soft_max->src[0];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(logits, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(weights, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(ids, &bindings[2])) {
        GGML_LOG_ERROR("%s: TOPK_MOE tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float scale = 1.0f;
    std::memcpy(&scale, reinterpret_cast<const int32_t *>(soft_max->op_params), sizeof(float));
    float clamp_min = -std::numeric_limits<float>::infinity();
    float clamp_max = std::numeric_limits<float>::infinity();
    if (clamp) {
        std::memcpy(&clamp_min, reinterpret_cast<const int32_t *>(clamp->op_params), sizeof(float));
        std::memcpy(&clamp_max, reinterpret_cast<const int32_t *>(clamp->op_params) + 1, sizeof(float));
    }

    const int64_t n_rows = ggml_nrows(logits);
    const int64_t n_expert_used = ggml_nelements(weights) / n_rows;
    ggml_backend_hrx_topk_moe_f32_constants constants = {
        /* .n_experts     = */ logits->ne[0],
        /* .n_rows        = */ n_rows,
        /* .n_expert_used = */ n_expert_used,
        /* .logits_nb1    = */ static_cast<int64_t>(logits->nb[1]),
        /* .weights_nb1   = */ ggml_backend_hrx_topk_moe_row_stride(weights, n_expert_used),
        /* .weights_nb_k  = */ ggml_backend_hrx_topk_moe_k_stride(weights, n_expert_used),
        /* .ids_nb1       = */ ggml_backend_hrx_topk_moe_row_stride(ids, n_expert_used),
        /* .ids_nb_k      = */ ggml_backend_hrx_topk_moe_k_stride(ids, n_expert_used),
        /* .scale         = */ scale,
        /* .clamp_min     = */ clamp_min,
        /* .clamp_max     = */ clamp_max,
        /* .with_norm     = */ clamp ? 1 : 0,
    };

    const auto & provider = ggml_backend_hrx_select_topk_moe_f32_provider(
        context->device_context, constants.n_rows);
    const uint32_t workgroup_size_x = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 64;
    const uint32_t workgroup_size_y = provider.export_info.workgroup_size[1] ?
        provider.export_info.workgroup_size[1] : 1;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.n_rows + workgroup_size_y - 1) / workgroup_size_y),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size_x, workgroup_size_y, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rope_f32(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: ROPE tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&freq_base, reinterpret_cast<const int32_t *>(dst->op_params) + 5, sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const int32_t *>(dst->op_params) + 6, sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const int32_t *>(dst->op_params) + 8, sizeof(float));

    ggml_backend_hrx_rope_f32_constants constants = {
        /* .ne00        = */ src0->ne[0],
        /* .ne01        = */ src0->ne[1],
        /* .ne02        = */ src0->ne[2],
        /* .nrows       = */ ggml_nrows(src0),
        /* .src_s1      = */ static_cast<int64_t>(src0->nb[1] / sizeof(float)),
        /* .src_s2      = */ static_cast<int64_t>(src0->nb[2] / sizeof(float)),
        /* .src_s3      = */ static_cast<int64_t>(src0->nb[3] / sizeof(float)),
        /* .dst_s1      = */ static_cast<int64_t>(dst->nb[1] / sizeof(float)),
        /* .dst_s2      = */ static_cast<int64_t>(dst->nb[2] / sizeof(float)),
        /* .dst_s3      = */ static_cast<int64_t>(dst->nb[3] / sizeof(float)),
        /* .n_dims      = */ ggml_get_op_params_i32(dst, 1),
        /* .mode        = */ ggml_get_op_params_i32(dst, 2),
        /* .section0    = */ ggml_get_op_params_i32(dst, 11),
        /* .section1    = */ ggml_get_op_params_i32(dst, 12),
        /* .section2    = */ ggml_get_op_params_i32(dst, 13),
        /* .section3    = */ ggml_get_op_params_i32(dst, 14),
        /* .freq_base   = */ freq_base,
        /* .freq_scale  = */ freq_scale,
        /* .attn_factor = */ attn_factor,
        /* ._pad        = */ 0.0f,
    };

    const auto & provider = context->device_context->rope_f32_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    const uint64_t total_pairs =
        static_cast<uint64_t>(constants.nrows) * static_cast<uint64_t>(constants.ne00 / 2);
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_rope_set_rows_f32_f16(
        ggml_backend_hrx_context * context,
        const ggml_tensor * rope,
        const ggml_tensor * set_rows) {
    const ggml_tensor * src0 = rope->src[0];
    const ggml_tensor * src1 = rope->src[1];
    const ggml_tensor * idxs = set_rows->src[1];
    hrx_buffer_ref_t bindings[4] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(idxs, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(set_rows, &bindings[3])) {
        GGML_LOG_ERROR("%s: ROPE_SET_ROWS tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float attn_factor = 0.0f;
    std::memcpy(&freq_base, reinterpret_cast<const int32_t *>(rope->op_params) + 5, sizeof(float));
    std::memcpy(&freq_scale, reinterpret_cast<const int32_t *>(rope->op_params) + 6, sizeof(float));
    std::memcpy(&attn_factor, reinterpret_cast<const int32_t *>(rope->op_params) + 8, sizeof(float));

    ggml_backend_hrx_rope_set_rows_f32_f16_constants constants = {
        /* .ne00             = */ src0->ne[0],
        /* .ne01             = */ src0->ne[1],
        /* .ne02             = */ src0->ne[2],
        /* .nrows            = */ ggml_nrows(src0),
        /* .src_s1           = */ static_cast<int64_t>(src0->nb[1] / sizeof(float)),
        /* .src_s2           = */ static_cast<int64_t>(src0->nb[2] / sizeof(float)),
        /* .src_s3           = */ static_cast<int64_t>(src0->nb[3] / sizeof(float)),
        /* .dst_s1           = */ static_cast<int64_t>(rope->nb[1] / sizeof(float)),
        /* .dst_s2           = */ static_cast<int64_t>(rope->nb[2] / sizeof(float)),
        /* .dst_s3           = */ static_cast<int64_t>(rope->nb[3] / sizeof(float)),
        /* .n_dims           = */ ggml_get_op_params_i32(rope, 1),
        /* .mode             = */ ggml_get_op_params_i32(rope, 2),
        /* .section0         = */ ggml_get_op_params_i32(rope, 11),
        /* .section1         = */ ggml_get_op_params_i32(rope, 12),
        /* .section2         = */ ggml_get_op_params_i32(rope, 13),
        /* .section3         = */ ggml_get_op_params_i32(rope, 14),
        /* .freq_base        = */ freq_base,
        /* .freq_scale       = */ freq_scale,
        /* .attn_factor      = */ attn_factor,
        /* ._pad             = */ 0.0f,
        /* .set_rows_ne1     = */ set_rows->ne[1],
        /* .set_rows_ne11    = */ idxs->ne[1],
        /* .set_rows_ne12    = */ idxs->ne[2],
        /* .idx_nb0          = */ static_cast<int64_t>(idxs->nb[0]),
        /* .idx_nb1          = */ static_cast<int64_t>(idxs->nb[1]),
        /* .idx_nb2          = */ static_cast<int64_t>(idxs->nb[2]),
        /* .set_rows_dst_nb1 = */ static_cast<int64_t>(set_rows->nb[1]),
        /* .set_rows_dst_nb2 = */ static_cast<int64_t>(set_rows->nb[2]),
        /* .set_rows_dst_nb3 = */ static_cast<int64_t>(set_rows->nb[3]),
    };

    const auto & provider = context->device_context->rope_set_rows_f32_f16_provider;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    const uint64_t total_pairs = static_cast<uint64_t>(constants.nrows) *
        static_cast<uint64_t>(constants.ne00 / 2);
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total_pairs + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            4,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_ssm_conv(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    hrx_buffer_ref_t bindings[3] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(src0, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(src1, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[2])) {
        GGML_LOG_ERROR("%s: SSM_CONV tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_ssm_conv_constants constants = {
        /* .d_conv     = */ src1->ne[0],
        /* .conv_width = */ src0->ne[0],
        /* .d_inner    = */ src0->ne[1],
        /* .n_tokens   = */ dst->ne[1],
        /* .n_seqs     = */ dst->ne[2],
        /* .src0_nb1   = */ static_cast<int64_t>(src0->nb[1]),
        /* .src0_nb2   = */ static_cast<int64_t>(src0->nb[2]),
        /* .weight_nb1 = */ static_cast<int64_t>(src1->nb[1]),
        /* .dst_nb1    = */ static_cast<int64_t>(dst->nb[1]),
        /* .dst_nb2    = */ static_cast<int64_t>(dst->nb[2]),
        /* .apply_silu = */ 0,
        /* .pad        = */ 0,
    };

    const auto & provider = context->device_context->ssm_conv_provider;
    const uint32_t workgroup_size_x = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 32;
    const uint32_t workgroup_size_y = provider.export_info.workgroup_size[1] ?
        provider.export_info.workgroup_size[1] : 16;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.d_inner + workgroup_size_x - 1) / workgroup_size_x),
            static_cast<uint32_t>((constants.n_tokens + workgroup_size_y - 1) / workgroup_size_y),
            static_cast<uint32_t>(constants.n_seqs),
        },
        /* .workgroup_size = */ { workgroup_size_x, workgroup_size_y, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 3, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_ssm_conv_update(
        ggml_backend_hrx_context * context,
        const ggml_tensor * concat,
        const ggml_tensor * state_update,
        const ggml_tensor * ssm,
        const ggml_tensor * fused_dst,
        bool apply_silu) {
    const ggml_tensor * conv_state = concat->src[0];
    const ggml_tensor * input = concat->src[1];
    const ggml_tensor * weight = ssm->src[1];
    const ggml_tensor * out = fused_dst ? fused_dst : ssm;
    hrx_buffer_ref_t bindings[5] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(conv_state, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(input, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(weight, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(state_update, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(out, &bindings[4])) {
        GGML_LOG_ERROR("%s: SSM_CONV_UPDATE tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    ggml_backend_hrx_ssm_conv_update_constants constants = {
        /* .d_conv           = */ weight->ne[0],
        /* .conv_state_width = */ conv_state->ne[0],
        /* .d_inner          = */ conv_state->ne[1],
        /* .n_tokens         = */ ssm->ne[1],
        /* .n_seqs           = */ ssm->ne[2],
        /* .state_nb0        = */ static_cast<int64_t>(conv_state->nb[0]),
        /* .state_nb1        = */ static_cast<int64_t>(conv_state->nb[1]),
        /* .state_nb2        = */ static_cast<int64_t>(conv_state->nb[2]),
        /* .input_nb0        = */ static_cast<int64_t>(input->nb[0]),
        /* .input_nb1        = */ static_cast<int64_t>(input->nb[1]),
        /* .weight_nb1       = */ static_cast<int64_t>(weight->nb[1]),
        /* .dst_nb1          = */ static_cast<int64_t>(out->nb[1]),
        /* .dst_nb2          = */ static_cast<int64_t>(out->nb[2]),
        /* .apply_silu       = */ apply_silu ? 1 : 0,
        /* .pad              = */ 0,
    };

    const auto & provider = context->device_context->ssm_conv_update_provider;
    const int64_t total = constants.d_inner * constants.n_tokens * constants.n_seqs;
    const uint32_t workgroup_size = provider.export_info.workgroup_size[0] ?
        provider.export_info.workgroup_size[0] : 256;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((total + workgroup_size - 1) / workgroup_size),
            1,
            1,
        },
        /* .workgroup_size = */ { workgroup_size, 1, 1 },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream,
            provider.executable,
            provider.export_ordinal,
            &config,
            &constants,
            sizeof(constants),
            bindings,
            5,
            HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }

    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_gated_delta_net(
        ggml_backend_hrx_context * context,
        const ggml_tensor * dst) {
    const ggml_tensor * q = dst->src[0];
    const ggml_tensor * k = dst->src[1];
    const ggml_tensor * v = dst->src[2];
    const ggml_tensor * g = dst->src[3];
    const ggml_tensor * beta = dst->src[4];
    const ggml_tensor * state = dst->src[5];
    hrx_buffer_ref_t bindings[8] = {};
    if (!ggml_backend_hrx_tensor_buffer_ref(q, &bindings[0]) ||
        !ggml_backend_hrx_tensor_buffer_ref(k, &bindings[1]) ||
        !ggml_backend_hrx_tensor_buffer_ref(v, &bindings[2]) ||
        !ggml_backend_hrx_tensor_buffer_ref(g, &bindings[3]) ||
        !ggml_backend_hrx_tensor_buffer_ref(beta, &bindings[4]) ||
        !ggml_backend_hrx_tensor_buffer_ref(state, &bindings[5]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[6]) ||
        !ggml_backend_hrx_tensor_buffer_ref(dst, &bindings[7])) {
        GGML_LOG_ERROR("%s: GATED_DELTA_NET tensor is not backed by a HRX buffer\n", __func__);
        return GGML_STATUS_FAILED;
    }

    const int64_t attn_score_elems = v->ne[0] * v->ne[1] * v->ne[2] * v->ne[3];
    ggml_backend_hrx_gated_delta_net_constants constants = {
        /* .S_v      = */ v->ne[0],
        /* .H        = */ v->ne[1],
        /* .n_tokens = */ v->ne[2],
        /* .n_seqs   = */ v->ne[3],
        /* .neq1     = */ q->ne[1],
        /* .nek1     = */ k->ne[1],
        /* .rq3      = */ v->ne[3] / q->ne[3],
        /* .rk3      = */ v->ne[3] / k->ne[3],
        /* .q_nb1    = */ static_cast<int64_t>(q->nb[1]),
        /* .q_nb2    = */ static_cast<int64_t>(q->nb[2]),
        /* .q_nb3    = */ static_cast<int64_t>(q->nb[3]),
        /* .k_nb1    = */ static_cast<int64_t>(k->nb[1]),
        /* .k_nb2    = */ static_cast<int64_t>(k->nb[2]),
        /* .k_nb3    = */ static_cast<int64_t>(k->nb[3]),
        /* .v_nb1    = */ static_cast<int64_t>(v->nb[1]),
        /* .v_nb2    = */ static_cast<int64_t>(v->nb[2]),
        /* .v_nb3    = */ static_cast<int64_t>(v->nb[3]),
        /* .g_ne0    = */ g->ne[0],
        /* .g_nb1    = */ static_cast<int64_t>(g->nb[1]),
        /* .g_nb2    = */ static_cast<int64_t>(g->nb[2]),
        /* .g_nb3    = */ static_cast<int64_t>(g->nb[3]),
        /* .beta_nb1 = */ static_cast<int64_t>(beta->nb[1]),
        /* .beta_nb2 = */ static_cast<int64_t>(beta->nb[2]),
        /* .beta_nb3 = */ static_cast<int64_t>(beta->nb[3]),
        /* .state_dst_offset = */ attn_score_elems,
        /* .scale    = */ 1.0f / std::sqrt(static_cast<float>(v->ne[0])),
        /* ._pad     = */ 0,
    };

    const auto & provider = context->device_context->gated_delta_net_provider;
    hrx_dispatch_config_t config = {
        /* .workgroup_count = */ {
            static_cast<uint32_t>((constants.S_v + 3) / 4),
            static_cast<uint32_t>(constants.H),
            static_cast<uint32_t>(constants.n_seqs),
        },
        /* .workgroup_size = */ {
            provider.export_info.workgroup_size[0] ? provider.export_info.workgroup_size[0] : 128,
            1,
            1,
        },
        /* .subgroup_size = */ 0,
    };

    if (!GGML_HRX_CHECK(hrx_stream_dispatch(
            context->stream, provider.executable, provider.export_ordinal, &config,
            &constants, sizeof(constants), bindings, 8, HRX_DISPATCH_FLAG_NONE))) {
        return GGML_STATUS_FAILED;
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_status ggml_backend_hrx_dispatch_add(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    if (ggml_backend_hrx_supports_binary_elementwise(context->device_context->add_provider, dst)) {
        return ggml_backend_hrx_dispatch_binary_elementwise(context, dst, context->device_context->add_provider, "ADD");
    }
    return ggml_backend_hrx_dispatch_broadcast_elementwise(
        context, dst, context->device_context->add_broadcast_provider, "ADD", false);
}

static ggml_status ggml_backend_hrx_dispatch_mul(ggml_backend_hrx_context * context, const ggml_tensor * dst) {
    if (ggml_backend_hrx_supports_binary_elementwise(context->device_context->mul_provider, dst)) {
        return ggml_backend_hrx_dispatch_binary_elementwise(context, dst, context->device_context->mul_provider, "MUL");
    }
    return ggml_backend_hrx_dispatch_broadcast_elementwise(
        context, dst, context->device_context->mul_broadcast_provider, "MUL", false);
}

struct ggml_backend_hrx_topk_moe_fusion {
    const ggml_tensor * soft_max = nullptr;
    const ggml_tensor * weights = nullptr;
    const ggml_tensor * ids = nullptr;
    const ggml_tensor * clamp = nullptr;
    int last_idx = -1;
    int ids_idx = -1;
    int weights_idx = -1;
    std::vector<int> idxs;
    std::vector<ggml_op> ops;
};

static bool ggml_backend_hrx_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const uintptr_t a_start = reinterpret_cast<uintptr_t>(a->data);
    const uintptr_t a_end = a_start + ggml_nbytes(a);
    const uintptr_t b_start = reinterpret_cast<uintptr_t>(b->data);
    const uintptr_t b_end = b_start + ggml_nbytes(b);
    return (b_start <= a_start && a_start < b_end) ||
           (a_start <= b_start && b_start < a_end);
}

static bool ggml_backend_hrx_topk_moe_fusion_memory_safe(
        const ggml_cgraph * cgraph,
        const ggml_backend_hrx_topk_moe_fusion & fusion) {
    const ggml_tensor * logits = fusion.soft_max ? fusion.soft_max->src[0] : nullptr;
    if (logits && ggml_nrows(logits) == 1) {
        return true;
    }

    const int output_idxs[2] = { fusion.ids_idx, fusion.weights_idx };
    for (int output_idx : output_idxs) {
        const ggml_tensor * dst = cgraph->nodes[output_idx];
        for (size_t i = 0; i < fusion.idxs.size(); ++i) {
            const ggml_tensor * node = cgraph->nodes[fusion.idxs[i]];
            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = node->src[src_idx];
                if (!src || src->op == GGML_OP_NONE ||
                    !ggml_backend_hrx_tensors_overlap(dst, src)) {
                    continue;
                }

                bool elided_source = false;
                for (size_t prev = 0; prev < i; ++prev) {
                    if (cgraph->nodes[fusion.idxs[prev]] == src) {
                        elided_source = true;
                        break;
                    }
                }
                if (!elided_source) {
                    return false;
                }
            }
        }
    }

    return true;
}

static bool ggml_backend_hrx_try_topk_moe_fusion(
        const ggml_cgraph * cgraph,
        int start,
        const ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_topk_moe_fusion * fusion) {
    *fusion = {};
    if (start >= cgraph->n_nodes || cgraph->nodes[start]->op != GGML_OP_SOFT_MAX) {
        return false;
    }

    auto add = [&](int idx) {
        fusion->idxs.push_back(idx);
        fusion->ops.push_back(cgraph->nodes[idx]->op);
    };
    auto next = [&](int idx) -> const ggml_tensor * {
        return idx < cgraph->n_nodes ? cgraph->nodes[idx] : nullptr;
    };

    fusion->soft_max = cgraph->nodes[start];
    add(start);

    int idx = start + 1;
    const ggml_tensor * probs = fusion->soft_max;
    if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == probs) {
        probs = next(idx);
        add(idx++);
    }

    const ggml_tensor * argsort = next(idx);
    if (!argsort || argsort->op != GGML_OP_ARGSORT ||
        (argsort->src[0] != fusion->soft_max && argsort->src[0] != probs) ||
        ggml_get_op_params_i32(argsort, 0) != GGML_SORT_ORDER_DESC) {
        return false;
    }
    add(idx++);

    const ggml_tensor * ids = next(idx);
    if (!ids || ids->op != GGML_OP_VIEW || ids->src[0] != argsort) {
        return false;
    }
    fusion->ids = ids;
    fusion->ids_idx = idx;
    add(idx++);

    const ggml_tensor * selected = next(idx);
    if (!selected || selected->op != GGML_OP_GET_ROWS ||
        selected->src[0] != probs || selected->src[1] != ids) {
        return false;
    }
    add(idx++);

    const ggml_tensor * weights_base = selected;
    if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == weights_base) {
        weights_base = next(idx);
        add(idx++);
    }

    const ggml_tensor * sum_rows = next(idx);
    if (sum_rows && sum_rows->op == GGML_OP_SUM_ROWS && sum_rows->src[0] == weights_base) {
        add(idx++);
        const ggml_tensor * clamp = next(idx);
        if (!clamp || clamp->op != GGML_OP_CLAMP || clamp->src[0] != sum_rows) {
            return false;
        }
        fusion->clamp = clamp;
        add(idx++);

        const ggml_tensor * div = next(idx);
        if (!div || div->op != GGML_OP_DIV || div->src[0] != weights_base || div->src[1] != clamp) {
            return false;
        }
        weights_base = div;
        add(idx++);

        if (next(idx) && next(idx)->op == GGML_OP_RESHAPE && next(idx)->src[0] == weights_base) {
            weights_base = next(idx);
            add(idx++);
        }
    }

    fusion->weights = weights_base;
    fusion->weights_idx = fusion->idxs.back();
    fusion->last_idx = fusion->idxs.back();

    if (!ggml_backend_hrx_supports_topk_moe_f32(
            device_context, fusion->soft_max, fusion->weights, fusion->ids)) {
        return false;
    }

    const int outputs[2] = { fusion->ids_idx, fusion->weights_idx };
    return ggml_can_fuse_subgraph_ext(
        cgraph,
        fusion->idxs.data(),
        static_cast<int>(fusion->idxs.size()),
        fusion->ops.data(),
        outputs,
        2) &&
        ggml_backend_hrx_topk_moe_fusion_memory_safe(cgraph, *fusion);
}

struct ggml_backend_hrx_ssm_conv_update_fusion {
    const ggml_tensor * state_view = nullptr;
    const ggml_tensor * state_update = nullptr;
    const ggml_tensor * ssm = nullptr;
    const ggml_tensor * out = nullptr;
    int state_view_idx = -1;
    int state_update_idx = -1;
    int ssm_idx = -1;
    int out_idx = -1;
    int last_idx = -1;
    bool apply_silu = false;
};

static bool ggml_backend_hrx_try_ssm_conv_update_fusion(
        const ggml_cgraph * cgraph,
        int concat_idx,
        const ggml_backend_hrx_device_context * device_context,
        ggml_backend_hrx_ssm_conv_update_fusion * fusion) {
    *fusion = {};
    const ggml_tensor * concat = cgraph->nodes[concat_idx];
    if (!concat || concat->op != GGML_OP_CONCAT || concat_idx + 3 >= cgraph->n_nodes) {
        return false;
    }

    const int state_view_idx = concat_idx + 1;
    const int state_update_idx = concat_idx + 2;
    const int ssm_idx = concat_idx + 3;
    const ggml_tensor * state_view = cgraph->nodes[state_view_idx];
    const ggml_tensor * state_update = cgraph->nodes[state_update_idx];
    const ggml_tensor * ssm = cgraph->nodes[ssm_idx];
    if (!state_view ||
        state_view->op != GGML_OP_VIEW ||
        !state_update ||
        state_update->op != GGML_OP_CPY ||
        state_update->src[0] != state_view ||
        !ssm ||
        ssm->op != GGML_OP_SSM_CONV ||
        ssm->src[0] != concat) {
        return false;
    }

    const ggml_tensor * out = ssm;
    int out_idx = ssm_idx;
    bool apply_silu = false;
    if (out_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * maybe_silu = cgraph->nodes[out_idx + 1];
        if (maybe_silu &&
            maybe_silu->op == GGML_OP_UNARY &&
            ggml_get_unary_op(maybe_silu) == GGML_UNARY_OP_SILU &&
            maybe_silu->src[0] == ssm) {
            out = maybe_silu;
            out_idx++;
            apply_silu = true;
        }
    }

    if (!ggml_backend_hrx_supports_ssm_conv_update(
            device_context, concat, state_view, state_update, ssm, apply_silu ? out : nullptr)) {
        return false;
    }

    std::array<int, 5> idxs = { concat_idx, state_view_idx, state_update_idx, ssm_idx, out_idx };
    std::array<ggml_op, 5> ops = { GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_CPY, GGML_OP_SSM_CONV, GGML_OP_UNARY };
    int count = 5;
    int outputs[2] = { state_update_idx, out_idx };
    if (!apply_silu) {
        count = 4;
        outputs[1] = ssm_idx;
    }
    if (!ggml_can_fuse_subgraph_ext(cgraph, idxs.data(), count, ops.data(), outputs, 2)) {
        return false;
    }

    fusion->state_view = state_view;
    fusion->state_update = state_update;
    fusion->ssm = ssm;
    fusion->out = out;
    fusion->state_view_idx = state_view_idx;
    fusion->state_update_idx = state_update_idx;
    fusion->ssm_idx = ssm_idx;
    fusion->out_idx = out_idx;
    fusion->last_idx = out_idx;
    fusion->apply_silu = apply_silu;
    return true;
}

struct ggml_backend_hrx_active_graph_guard {
    ggml_backend_hrx_context * context = nullptr;
    ggml_backend_hrx_context * previous_context = nullptr;
    const ggml_tensor * previous_node = nullptr;

    explicit ggml_backend_hrx_active_graph_guard(ggml_backend_hrx_context * context)
        : context(context),
          previous_context(g_hrx_active_graph_context),
          previous_node(g_hrx_active_graph_node) {
        ggml_backend_hrx_begin_submit_batch(context);
        g_hrx_active_graph_context = context;
        g_hrx_active_graph_node = nullptr;
    }

    ~ggml_backend_hrx_active_graph_guard() {
        if (context) {
            context->last_total_mul_mat_bytes = context->total_mul_mat_bytes;
            if (ggml_backend_hrx_env_enabled("GGML_HRX_TRACE_SUBMIT_BATCHING")) {
                GGML_LOG_DEBUG(
                    "%s: graph complete total_mul_mat_bytes=%" PRIu64 " submit_flushes=%" PRIu64 "\n",
                    __func__,
                    context->last_total_mul_mat_bytes,
                    context->submit_flush_count);
            }
        }
        g_hrx_active_graph_context = previous_context;
        g_hrx_active_graph_node = previous_node;
    }
};

static ggml_status ggml_backend_hrx_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * context = static_cast<ggml_backend_hrx_context *>(backend->context);
    if (!ggml_backend_hrx_sync_streams(context->device_context)) {
        return GGML_STATUS_FAILED;
    }
    {
        std::lock_guard<std::mutex> lock(context->device_context->streams_mutex);
        context->device_context->active_stream = context->stream;
    }
    ggml_backend_hrx_active_graph_guard active_graph_guard(context);

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        const ggml_tensor * node = cgraph->nodes[i];
        g_hrx_active_graph_node = node;
        ggml_backend_hrx_topk_moe_fusion topk_moe;
        if (!ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            ggml_backend_hrx_try_topk_moe_fusion(cgraph, i, context->device_context, &topk_moe)) {
            if (ggml_backend_hrx_dispatch_topk_moe_f32(
                    context, topk_moe.soft_max, topk_moe.weights, topk_moe.ids, topk_moe.clamp) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i = topk_moe.last_idx;
            continue;
        }
        ggml_backend_hrx_ssm_conv_update_fusion ssm_update;
        if (node->op == GGML_OP_CONCAT &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            ggml_backend_hrx_try_ssm_conv_update_fusion(cgraph, i, context->device_context, &ssm_update)) {
            if (ggml_backend_hrx_dispatch_ssm_conv_update(
                    context,
                    node,
                    ssm_update.state_update,
                    ssm_update.ssm,
                    ssm_update.apply_silu ? ssm_update.out : nullptr,
                    ssm_update.apply_silu) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i = ssm_update.last_idx;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_RMS_NORM_MUL_ROPE_FUSION") &&
            i + 4 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 2]->op == GGML_OP_ROPE &&
            cgraph->nodes[i + 3]->op == GGML_OP_VIEW &&
            cgraph->nodes[i + 4]->op == GGML_OP_SET_ROWS &&
            ggml_backend_hrx_supports_rms_norm_mul_rope_set_rows_f32_f16(
                context->device_context,
                node,
                cgraph->nodes[i + 1],
                cgraph->nodes[i + 2],
                cgraph->nodes[i + 3],
                cgraph->nodes[i + 4]) &&
            ggml_can_fuse_subgraph(
                cgraph,
                i,
                { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS },
                { i + 4 })) {
            if (ggml_backend_hrx_dispatch_rms_norm_mul_rope_set_rows_f32_f16(
                    context, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 4;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_RMS_NORM_MUL_ROPE_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            cgraph->nodes[i + 2]->op == GGML_OP_ROPE &&
            ggml_backend_hrx_supports_rms_norm_mul_rope_f32(
                context->device_context, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(
                cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, { i + 2 })) {
            if (ggml_backend_hrx_dispatch_rms_norm_mul_rope_f32(
                    context, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        if (node->op == GGML_OP_RMS_NORM &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_RMS_NORM_MUL_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            ggml_backend_hrx_supports_rms_norm_mul(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, { i + 1 })) {
            if (ggml_backend_hrx_dispatch_rms_norm_mul(context, node, cgraph->nodes[i + 1]) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_MAT_SWIGLU_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT &&
            cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            const ggml_tensor * first = node;
            const ggml_tensor * second = cgraph->nodes[i + 1];
            const ggml_tensor * swiglu = cgraph->nodes[i + 2];
            const ggml_tensor * gate = swiglu->src[0];
            const ggml_tensor * up = swiglu->src[1];
            if (((gate == first && up == second) || (gate == second && up == first)) &&
                ggml_backend_hrx_supports_mul_mat_vec_bf16_swiglu(context->device_context, gate, up, swiglu) &&
                ggml_can_fuse_subgraph(
                    cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, { i + 2 })) {
                if (ggml_backend_hrx_dispatch_mul_mat_vec_bf16_swiglu(context, gate, up, swiglu) !=
                    GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                i += 2;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_MAT_ADD_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_ADD &&
            ggml_backend_hrx_supports_mul_mat_vec_q8_0_add(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_ADD }, { i + 1 })) {
            if (ggml_backend_hrx_dispatch_mul_mat_vec_q8_0_add(context, node, cgraph->nodes[i + 1]) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_MUL_MAT_ID &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_MAT_ID_SWIGLU_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL_MAT_ID &&
            cgraph->nodes[i + 2]->op == GGML_OP_GLU) {
            const ggml_tensor * first = node;
            const ggml_tensor * second = cgraph->nodes[i + 1];
            const ggml_tensor * swiglu = cgraph->nodes[i + 2];
            const ggml_tensor * gate = swiglu->src[0];
            const ggml_tensor * up = swiglu->src[1];
            if (((gate == first && up == second) || (gate == second && up == first)) &&
                ggml_backend_hrx_supports_mul_mat_id_q4_k_swiglu(context->device_context, gate, up, swiglu) &&
                ggml_can_fuse_subgraph(
                    cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU }, { i + 2 })) {
                if (ggml_backend_hrx_dispatch_mul_mat_id_q4_k_swiglu(context, gate, up, swiglu) !=
                    GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                i += 2;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL_MAT_ID &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_MUL_MAT_ID_MUL_FUSION") &&
            i + 1 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_MUL &&
            ggml_backend_hrx_supports_mul_mat_id_q4_k_mul(context->device_context, node, cgraph->nodes[i + 1]) &&
            ggml_can_fuse_subgraph(cgraph, i, { GGML_OP_MUL_MAT_ID, GGML_OP_MUL }, { i + 1 })) {
            const ggml_tensor * mul = cgraph->nodes[i + 1];
            const ggml_status status =
                ggml_backend_hrx_supports_mul_mat_id_q4_k_mul_q8_1(context->device_context, node, mul) ?
                    ggml_backend_hrx_dispatch_mul_mat_id_q4_k_mul_q8_1(context, node, mul) :
                    ggml_backend_hrx_dispatch_mul_mat_id_q4_k_mul(context, node, mul);
            if (status != GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i++;
            continue;
        }
        if (node->op == GGML_OP_MUL && !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION")) {
            std::array<const ggml_tensor *, 8> sources = {};
            const ggml_tensor * sum8 = nullptr;
            int last_idx = -1;
            if (ggml_backend_hrx_find_mul_sum8_fusion(
                    context->device_context, cgraph, i, &sources, &sum8, &last_idx)) {
                if (ggml_backend_hrx_dispatch_mul_sum8_f32(context, node, sum8) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                i = last_idx;
                continue;
            }
        }
        if (node->op == GGML_OP_MUL &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_ADD &&
            cgraph->nodes[i + 2]->op == GGML_OP_ADD) {
            const ggml_tensor * add_src0 = nullptr;
            const ggml_tensor * add_src1 = nullptr;
            const ggml_tensor * first_add = cgraph->nodes[i + 1];
            const ggml_tensor * second_add = cgraph->nodes[i + 2];
            if (ggml_backend_hrx_supports_mul_add_add_broadcast(
                    context->device_context, node, first_add, second_add, &add_src0, &add_src1) &&
                ggml_can_fuse_subgraph(
                    cgraph, i, { GGML_OP_MUL, GGML_OP_ADD, GGML_OP_ADD }, { i + 2 })) {
                if (ggml_backend_hrx_dispatch_mul_add_add_broadcast_f32(
                        context, node, second_add, add_src0, add_src1) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                i += 2;
                continue;
            }
        }
        if (node->op == GGML_OP_ADD &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ADD8_FUSION")) {
            std::array<const ggml_tensor *, 8> sources = {};
            const ggml_tensor * add8 = nullptr;
            if (ggml_backend_hrx_try_collect_add8_chain(
                    context->device_context, cgraph, i, &sources, &add8)) {
                if (ggml_backend_hrx_dispatch_add8_f32(context, sources, add8) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                i += 6;
                continue;
            }
        }
        if (node->op == GGML_OP_ROPE &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_FUSION") &&
            !ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ROPE_SET_ROWS_FUSION") &&
            i + 2 < cgraph->n_nodes &&
            cgraph->nodes[i + 1]->op == GGML_OP_VIEW &&
            cgraph->nodes[i + 2]->op == GGML_OP_SET_ROWS &&
            ggml_backend_hrx_supports_rope_set_rows_f32_f16(
                context->device_context, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]) &&
            ggml_can_fuse_subgraph(
                cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, { i + 2 })) {
            if (ggml_backend_hrx_dispatch_rope_set_rows_f32_f16(context, node, cgraph->nodes[i + 2]) !=
                GGML_STATUS_SUCCESS) {
                return GGML_STATUS_FAILED;
            }
            i += 2;
            continue;
        }
        switch (node->op) {
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                break;
            case GGML_OP_RMS_NORM:
                if (!ggml_backend_hrx_supports_rms_norm(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: RMS_NORM shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_rms_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ADD:
                if (!ggml_backend_hrx_supports_add(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ADD shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_add(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL:
                if (!ggml_backend_hrx_supports_mul(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_mul(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_DIV:
                if (!ggml_backend_hrx_supports_div(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: DIV shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_broadcast_elementwise(
                        context, node, context->device_context->div_broadcast_provider, "DIV", true) !=
                    GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SCALE:
                if (!ggml_backend_hrx_supports_scale(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SCALE shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_scale_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CPY:
                if (!ggml_backend_hrx_supports_cpy(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: CPY shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_cpy(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONT:
                if (!ggml_backend_hrx_supports_cont(node)) {
                    GGML_LOG_ERROR("%s: CONT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_cpy(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SET_ROWS:
                if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_SET_ROWS")) {
                    GGML_LOG_ERROR("%s: SET_ROWS disabled by GGML_HRX_DISABLE_SET_ROWS\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (const ggml_tensor * mul_mat = ggml_backend_hrx_unwrap_reshape_view_src0(node->src[0]);
                    ggml_backend_hrx_supports_mul_mat_vec_bf16_set_rows_f16(
                        context->device_context, mul_mat, node->src[0], node)) {
                    if (ggml_backend_hrx_dispatch_mul_mat_vec_bf16_set_rows_f16(context, mul_mat, node) !=
                        GGML_STATUS_SUCCESS) {
                        return GGML_STATUS_FAILED;
                    }
                    break;
                }
                if (!ggml_backend_hrx_supports_set_rows(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SET_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_set_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GET_ROWS:
                if (!ggml_backend_hrx_supports_get_rows_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GET_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_get_rows_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT:
                if (!ggml_backend_hrx_supports_mul_mat_vec(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_mul_mat_vec(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT:
                if (!ggml_backend_hrx_supports_flash_attn_ext_f32_decode(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: FLASH_ATTN_EXT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_flash_attn_ext_f32_decode(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CONCAT:
                if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_CONCAT")) {
                    GGML_LOG_ERROR("%s: CONCAT disabled by GGML_HRX_DISABLE_CONCAT\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (!ggml_backend_hrx_supports_concat_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: CONCAT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_concat_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SOFT_MAX:
                if (!ggml_backend_hrx_supports_soft_max_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SOFT_MAX shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_soft_max_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ARGSORT:
                if (!ggml_backend_hrx_supports_argsort_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ARGSORT shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_argsort_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_ROPE:
                if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_ROPE")) {
                    GGML_LOG_ERROR("%s: ROPE disabled by GGML_HRX_DISABLE_ROPE\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (!ggml_backend_hrx_supports_rope_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: ROPE shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_rope_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_UNARY: {
                if (!ggml_backend_hrx_supports_unary_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: UNARY shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                const ggml_backend_hrx_op_provider * provider =
                    ggml_backend_hrx_unary_provider(context->device_context, node);
                if (!provider ||
                    ggml_backend_hrx_dispatch_unary_f32(
                        context, node, *provider, ggml_unary_op_name(ggml_get_unary_op(node))) !=
                        GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            }
            case GGML_OP_GLU:
                if (!ggml_backend_hrx_supports_swiglu_f32(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GLU shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_swiglu_f32(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SUM_ROWS:
                if (!ggml_backend_hrx_supports_sum_rows(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SUM_ROWS shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_sum_rows(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_L2_NORM:
                if (!ggml_backend_hrx_supports_l2_norm(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: L2_NORM shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_l2_norm(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_CLAMP:
                if (!ggml_backend_hrx_supports_clamp(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: CLAMP shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_clamp(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_SSM_CONV:
                if (ggml_backend_hrx_env_enabled("GGML_HRX_DISABLE_SSM_CONV")) {
                    GGML_LOG_ERROR("%s: SSM_CONV disabled by GGML_HRX_DISABLE_SSM_CONV\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (!ggml_backend_hrx_supports_ssm_conv(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: SSM_CONV shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_ssm_conv(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_GATED_DELTA_NET:
                if (!ggml_backend_hrx_supports_gated_delta_net(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: GATED_DELTA_NET shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if (ggml_backend_hrx_dispatch_gated_delta_net(context, node) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            case GGML_OP_MUL_MAT_ID:
                if (!ggml_backend_hrx_supports_mul_mat_id_q4_k(context->device_context, node)) {
                    GGML_LOG_ERROR("%s: MUL_MAT_ID shape/type/layout is unsupported\n", __func__);
                    return GGML_STATUS_FAILED;
                }
                if ((ggml_backend_hrx_supports_mul_mat_id_q4_k_q8_1(context->device_context, node) ?
                        ggml_backend_hrx_dispatch_mul_mat_id_q4_k_q8_1(context, node) :
                        ggml_backend_hrx_dispatch_mul_mat_id_q4_k(context, node)) != GGML_STATUS_SUCCESS) {
                    return GGML_STATUS_FAILED;
                }
                break;
            default:
                GGML_LOG_ERROR("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
                return GGML_STATUS_FAILED;
        }
    }

    ggml_backend_hrx_synchronize(backend);
    return GGML_STATUS_SUCCESS;
}

static const ggml_backend_i ggml_backend_hrx_i = {
    /* .get_name           = */ ggml_backend_hrx_get_name,
    /* .free               = */ ggml_backend_hrx_free,
    /* .set_tensor_async   = */ nullptr,
    /* .get_tensor_async   = */ nullptr,
    /* .cpy_tensor_async   = */ nullptr,
    /* .synchronize        = */ ggml_backend_hrx_synchronize,
    /* .graph_plan_create  = */ nullptr,
    /* .graph_plan_free    = */ nullptr,
    /* .graph_plan_update  = */ nullptr,
    /* .graph_plan_compute = */ nullptr,
    /* .graph_compute      = */ ggml_backend_hrx_graph_compute,
    /* .event_record       = */ nullptr,
    /* .event_wait         = */ nullptr,
    /* .graph_optimize     = */ nullptr,
};

static const char * ggml_backend_hrx_device_get_name(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->name.c_str();
}

static const char * ggml_backend_hrx_device_get_description(ggml_backend_dev_t dev) {
    return ggml_backend_hrx_get_device_context(dev)->description.c_str();
}

static void ggml_backend_hrx_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    auto * context = ggml_backend_hrx_get_device_context(dev);
    *free = context->memory_total;
    *total = context->memory_total;
}

static enum ggml_backend_dev_type ggml_backend_hrx_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_hrx_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_hrx_device_get_name(dev);
    props->description = ggml_backend_hrx_device_get_description(dev);
    props->type = ggml_backend_hrx_device_get_type(dev);
    ggml_backend_hrx_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->device_id = nullptr;
    props->caps = {
        /* .async = */ true,
        /* .host_buffer = */ false,
        /* .buffer_from_host_ptr = */ false,
        /* .events = */ false,
    };
}

static ggml_backend_t ggml_backend_hrx_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);

    auto * device_context = ggml_backend_hrx_get_device_context(dev);
    hrx_stream_t stream = nullptr;
    if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &stream))) {
        return nullptr;
    }

    auto * context = new (std::nothrow) ggml_backend_hrx_context {
        /* .device_context  = */ device_context,
        /* .stream          = */ stream,
        /* .name            = */ device_context->name,
        /* .scratch_buffers = */ {},
    };
    if (!context) {
        hrx_stream_release(stream);
        return nullptr;
    }

    ggml_backend_t backend = new (std::nothrow) ggml_backend {
        /* .guid    = */ ggml_backend_hrx_guid(),
        /* .iface   = */ ggml_backend_hrx_i,
        /* .device  = */ dev,
        /* .context = */ context,
    };
    if (!backend) {
        hrx_stream_release(stream);
        delete context;
        return nullptr;
    }

    ggml_backend_hrx_register_stream(device_context, stream);
    return backend;
}

static bool ggml_backend_hrx_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);
    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;
        case GGML_OP_RMS_NORM:
            return ggml_backend_hrx_supports_rms_norm(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_ADD:
            return ggml_backend_hrx_supports_add(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_MUL:
            return ggml_backend_hrx_supports_mul(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_DIV:
            return ggml_backend_hrx_supports_div(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_SCALE:
            return ggml_backend_hrx_supports_scale(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_CPY:
            return ggml_backend_hrx_supports_cpy(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_CONT:
            return ggml_backend_hrx_supports_cont(op);
        case GGML_OP_SET_ROWS:
            return ggml_backend_hrx_supports_set_rows(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_GET_ROWS:
            return ggml_backend_hrx_supports_get_rows_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_MUL_MAT:
            return ggml_backend_hrx_supports_mul_mat_vec(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_backend_hrx_supports_flash_attn_ext_f32_decode(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_CONCAT:
            return ggml_backend_hrx_supports_concat_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_SOFT_MAX:
            return ggml_backend_hrx_supports_soft_max_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_ARGSORT:
            return ggml_backend_hrx_supports_argsort_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_ROPE:
            return ggml_backend_hrx_supports_rope_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_UNARY:
            return ggml_backend_hrx_supports_unary_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_GLU:
            return ggml_backend_hrx_supports_swiglu_f32(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_SUM_ROWS:
            return ggml_backend_hrx_supports_sum_rows(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_L2_NORM:
            return ggml_backend_hrx_supports_l2_norm(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_CLAMP:
            return ggml_backend_hrx_supports_clamp(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_SSM_CONV:
            return ggml_backend_hrx_supports_ssm_conv(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_GATED_DELTA_NET:
            return ggml_backend_hrx_supports_gated_delta_net(ggml_backend_hrx_get_device_context(dev), op);
        case GGML_OP_MUL_MAT_ID:
            return ggml_backend_hrx_supports_mul_mat_id_q4_k(ggml_backend_hrx_get_device_context(dev), op);
        default:
            return false;
    }
}

static bool ggml_backend_hrx_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_hrx_buffer_type_get_name) {
        return false;
    }
    return buft->device == dev;
}

static const ggml_backend_device_i ggml_backend_hrx_device_i = {
    /* .get_name             = */ ggml_backend_hrx_device_get_name,
    /* .get_description      = */ ggml_backend_hrx_device_get_description,
    /* .get_memory           = */ ggml_backend_hrx_device_get_memory,
    /* .get_type             = */ ggml_backend_hrx_device_get_type,
    /* .get_props            = */ ggml_backend_hrx_device_get_props,
    /* .init_backend         = */ ggml_backend_hrx_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_hrx_device_buffer_type,
    /* .get_host_buffer_type = */ nullptr,
    /* .buffer_from_host_ptr = */ nullptr,
    /* .supports_op          = */ ggml_backend_hrx_device_supports_op,
    /* .supports_buft        = */ ggml_backend_hrx_device_supports_buft,
    /* .offload_op           = */ nullptr,
    /* .event_new            = */ nullptr,
    /* .event_free           = */ nullptr,
    /* .event_synchronize    = */ nullptr,
};

static ggml_backend_hrx_reg_context * ggml_backend_hrx_get_reg_context(ggml_backend_reg_t reg) {
    return static_cast<ggml_backend_hrx_reg_context *>(reg->context);
}

static const char * ggml_backend_hrx_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_HRX_NAME;
}

static size_t ggml_backend_hrx_reg_get_device_count(ggml_backend_reg_t reg) {
    return ggml_backend_hrx_get_reg_context(reg)->devices.size();
}

static ggml_backend_dev_t ggml_backend_hrx_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    auto * context = ggml_backend_hrx_get_reg_context(reg);
    GGML_ASSERT(index < context->devices.size());
    return &context->devices[index];
}

static void * ggml_backend_hrx_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_hrx_reg_i = {
    /* .get_name         = */ ggml_backend_hrx_reg_get_name,
    /* .get_device_count = */ ggml_backend_hrx_reg_get_device_count,
    /* .get_device       = */ ggml_backend_hrx_reg_get_device,
    /* .get_proc_address = */ ggml_backend_hrx_reg_get_proc_address,
};

static std::unique_ptr<ggml_backend_hrx_reg_context> ggml_backend_hrx_create_reg_context() {
    auto context = std::make_unique<ggml_backend_hrx_reg_context>();

    hrx_status_t status = hrx_gpu_initialize(0);
    if (hrx_status_is_ok(status)) {
        context->gpu_initialized = true;
    } else if (hrx_status_code(status) == HRX_STATUS_ALREADY_EXISTS) {
        hrx_status_ignore(status);
    } else {
        hrx_status_ignore(status);
        return context;
    }

    int device_count = 0;
    if (!GGML_HRX_CHECK(hrx_gpu_device_count(&device_count)) || device_count <= 0) {
        return context;
    }

    context->device_contexts.reserve(device_count);
    context->devices.reserve(device_count);

    for (int i = 0; i < device_count; ++i) {
        hrx_device_t device = nullptr;
        if (!GGML_HRX_CHECK(hrx_gpu_device_get(i, &device)) || !device) {
            continue;
        }
        hrx_device_retain(device);

        auto device_context = std::make_unique<ggml_backend_hrx_device_context>();
        device_context->device = device;
        device_context->name = std::string(GGML_HRX_NAME) + std::to_string(i);
        device_context->description = ggml_backend_hrx_device_description(device);
        device_context->architecture = ggml_backend_hrx_device_architecture(device);
        device_context->memory_total = ggml_backend_hrx_total_memory(device);
        if (!GGML_HRX_CHECK(hrx_stream_create(device_context->device, 0, &device_context->transfer_stream))) {
            hrx_device_release(device);
            continue;
        }
        ggml_backend_hrx_register_stream(device_context.get(), device_context->transfer_stream);
        (void) ggml_backend_hrx_load_rms_norm_provider(device_context.get());
        (void) ggml_backend_hrx_load_rms_norm_mul_providers(device_context.get());
        (void) ggml_backend_hrx_load_add_provider(device_context.get());
        (void) ggml_backend_hrx_load_mul_provider(device_context.get());
        (void) ggml_backend_hrx_load_add_broadcast_provider(device_context.get());
        (void) ggml_backend_hrx_load_mul_broadcast_provider(device_context.get());
        (void) ggml_backend_hrx_load_div_broadcast_provider(device_context.get());
        (void) ggml_backend_hrx_load_add8_provider(device_context.get());
        (void) ggml_backend_hrx_load_mul_sum8_provider(device_context.get());
        (void) ggml_backend_hrx_load_mul_add_add_broadcast_provider(device_context.get());
        (void) ggml_backend_hrx_load_scale_provider(device_context.get());
        (void) ggml_backend_hrx_load_set_rows_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_set_rows_f16_provider(device_context.get());
        (void) ggml_backend_hrx_load_set_rows_q8_0_provider(device_context.get());
        (void) ggml_backend_hrx_load_set_rows_q4_0_provider(device_context.get());
        (void) ggml_backend_hrx_load_silu_provider(device_context.get());
        (void) ggml_backend_hrx_load_sigmoid_provider(device_context.get());
        (void) ggml_backend_hrx_load_softplus_provider(device_context.get());
        (void) ggml_backend_hrx_load_swiglu_provider(device_context.get());
        (void) ggml_backend_hrx_load_sum_rows_provider(device_context.get());
        (void) ggml_backend_hrx_load_l2_norm_provider(device_context.get());
        (void) ggml_backend_hrx_load_clamp_provider(device_context.get());
        (void) ggml_backend_hrx_load_get_rows_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_get_rows_q5_k_provider(device_context.get());
        (void) ggml_backend_hrx_load_mul_mat_vec_providers(device_context.get());
        (void) ggml_backend_hrx_load_mul_mat_id_providers(device_context.get());
        (void) ggml_backend_hrx_load_flash_attn_ext_providers(device_context.get());
        (void) ggml_backend_hrx_load_concat_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_copy_strided_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_copy_f32_f16_provider(device_context.get());
        (void) ggml_backend_hrx_load_soft_max_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_soft_max_f32_mask_provider(device_context.get());
        (void) ggml_backend_hrx_load_argsort_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_topk_moe_f32_providers(device_context.get());
        (void) ggml_backend_hrx_load_rope_f32_provider(device_context.get());
        (void) ggml_backend_hrx_load_rope_set_rows_f32_f16_provider(device_context.get());
        (void) ggml_backend_hrx_load_ssm_conv_provider(device_context.get());
        (void) ggml_backend_hrx_load_ssm_conv_update_provider(device_context.get());
        (void) ggml_backend_hrx_load_gated_delta_net_provider(device_context.get());

        context->device_contexts.emplace_back(std::move(device_context));
        context->devices.push_back({
            /* .iface   = */ ggml_backend_hrx_device_i,
            /* .reg     = */ nullptr,
            /* .context = */ context->device_contexts.back().get(),
        });
    }

    return context;
}

} // namespace

ggml_backend_t ggml_backend_hrx_init(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        GGML_LOG_ERROR("%s: invalid HRX device index %zu\n", __func__, dev_num);
        return nullptr;
    }
    return ggml_backend_dev_init(ggml_backend_reg_dev_get(reg, dev_num), nullptr);
}

bool ggml_backend_is_hrx(ggml_backend_t backend) {
    return backend != nullptr && ggml_guid_matches(backend->guid, ggml_backend_hrx_guid());
}

int ggml_backend_hrx_get_device_count(void) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    return reg ? static_cast<int>(ggml_backend_reg_dev_count(reg)) : 0;
}

void ggml_backend_hrx_get_device_description(int device, char * description, size_t description_size) {
    if (!description || description_size == 0) {
        return;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        description[0] = '\0';
        return;
    }

    const char * value = ggml_backend_dev_description(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)));
    std::snprintf(description, description_size, "%s", value ? value : "");
}

void ggml_backend_hrx_get_device_memory(int device, size_t * free, size_t * total) {
    if (free) {
        *free = 0;
    }
    if (total) {
        *total = 0;
    }

    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || device < 0 || static_cast<size_t>(device) >= ggml_backend_reg_dev_count(reg)) {
        return;
    }

    ggml_backend_dev_memory(
        ggml_backend_reg_dev_get(reg, static_cast<size_t>(device)), free, total);
}

ggml_backend_buffer_type_t ggml_backend_hrx_buffer_type(size_t dev_num) {
    ggml_backend_reg_t reg = ggml_backend_hrx_reg();
    if (!reg || dev_num >= ggml_backend_reg_dev_count(reg)) {
        return nullptr;
    }
    return ggml_backend_dev_buffer_type(ggml_backend_reg_dev_get(reg, dev_num));
}

ggml_backend_reg_t ggml_backend_hrx_reg(void) {
    static std::unique_ptr<ggml_backend_hrx_reg_context> context =
        ggml_backend_hrx_create_reg_context();

    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_hrx_reg_i,
        /* .context     = */ context.get(),
    };

    if (context) {
        for (auto & device : context->devices) {
            device.reg = &reg;
        }
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_hrx_reg)
