#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_gated_delta_net_f32_constants {
    long long S_v;
    long long H;
    long long n_tokens;
    long long n_seqs;
    long long neq1;
    long long nek1;
    long long rq3;
    long long rk3;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long g_ne0;
    long long g_nb1;
    long long g_nb2;
    long long g_nb3;
    long long beta_nb1;
    long long beta_nb2;
    long long beta_nb3;
    long long state_dst_offset;
    float scale;
    int pad;
};

struct hrx_gated_delta_net_s128_nokda_nomod_constants {
    long long H;
    long long n_tokens;
    long long n_seqs;
    long long q_head_mask;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_head_mask;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long g_nb1;
    long long g_nb2;
    long long g_nb3;
    long long beta_nb1;
    long long beta_nb2;
    long long beta_nb3;
    long long state_dst_offset;
    float scale;
    int pad;
};

static_assert(sizeof(hrx_gated_delta_net_s128_nokda_nomod_constants) == 176);

struct hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_constants {
    long long state_dst_offset;
    float scale;
    int pad;
};

static_assert(sizeof(hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_constants) == 16);

static __device__ __forceinline__ float hrx_sigmoid_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void hrx_gated_delta_net_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_f32_constants c) {
    constexpr unsigned int lanes_per_column = 32;
    constexpr unsigned int columns_per_workgroup = 4;
    constexpr unsigned int max_rows_per_lane = 8; // supports S_v <= 256

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const long long col =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group);
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();

    if (head >= c.H || seq >= c.n_seqs || col_group >= columns_per_workgroup) {
        return;
    }

    const bool active_col = col < c.S_v;
    const long long safe_col = active_col ? col : 0;
    __shared__ float reduce[lanes_per_column * columns_per_workgroup];

    const long long iq1 = head % c.neq1;
    const long long ik1 = head % c.nek1;
    const long long iq3 = seq / c.rq3;
    const long long ik3 = seq / c.rk3;
    const bool kda = c.g_ne0 == c.S_v;

    const long long attn_score_elems = c.S_v * c.H * c.n_tokens * c.n_seqs;
    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * c.S_v + safe_col;
    (void) attn_score_elems;
    float * state_out = state_dst + c.state_dst_offset + (seq * c.H + head) * c.S_v * c.S_v + safe_col * c.S_v;
    const float * state_col = state_in + (seq * c.H + head) * c.S_v * c.S_v + safe_col * c.S_v;

    float s_shard[max_rows_per_lane];
    for (unsigned int r = 0; r < max_rows_per_lane; ++r) {
        const unsigned int row = r * lanes_per_column + lane;
        s_shard[r] = (active_col && row < static_cast<unsigned int>(c.S_v)) ? state_col[row] : 0.0f;
    }

    for (long long token = 0; token < c.n_tokens; ++token) {
        const char * q_base = reinterpret_cast<const char *>(q) + iq3 * c.q_nb3 + token * c.q_nb2 + iq1 * c.q_nb1;
        const char * k_base = reinterpret_cast<const char *>(k) + ik3 * c.k_nb3 + token * c.k_nb2 + ik1 * c.k_nb1;
        const char * v_base = reinterpret_cast<const char *>(v) + seq * c.v_nb3 + token * c.v_nb2 + head * c.v_nb1;
        const char * g_base = reinterpret_cast<const char *>(g) + seq * c.g_nb3 + token * c.g_nb2 + head * c.g_nb1;
        const char * beta_base =
            reinterpret_cast<const char *>(beta) + seq * c.beta_nb3 + token * c.beta_nb2 + head * c.beta_nb1;

        float q_reg[max_rows_per_lane];
        float k_reg[max_rows_per_lane];
        float g_reg[max_rows_per_lane];
        for (unsigned int r = 0; r < max_rows_per_lane; ++r) {
            const unsigned int row = r * lanes_per_column + lane;
            const bool active_row = active_col && row < static_cast<unsigned int>(c.S_v);
            q_reg[r] = active_row ? *reinterpret_cast<const float *>(q_base + row * sizeof(float)) : 0.0f;
            k_reg[r] = active_row ? *reinterpret_cast<const float *>(k_base + row * sizeof(float)) : 0.0f;
            g_reg[r] = active_row ?
                (kda ? __builtin_expf(*reinterpret_cast<const float *>(g_base + row * sizeof(float))) : 1.0f) :
                0.0f;
        }

        float kv_partial = 0.0f;
        for (unsigned int r = 0; r < max_rows_per_lane; ++r) {
            kv_partial += g_reg[r] * s_shard[r] * k_reg[r];
        }
        reduce[tid] = kv_partial;
        __builtin_amdgcn_s_barrier();

        for (unsigned int step = lanes_per_column / 2; step > 0; step >>= 1) {
            if (lane < step) {
                reduce[tid] += reduce[tid + step];
            }
            __builtin_amdgcn_s_barrier();
        }

        const float kv_col = reduce[col_group * lanes_per_column];
        const float beta_val = *reinterpret_cast<const float *>(beta_base);
        const float v_col = active_col ? *reinterpret_cast<const float *>(v_base + col * sizeof(float)) : 0.0f;
        const float g_scalar = kda ? 1.0f : __builtin_expf(*reinterpret_cast<const float *>(g_base));
        const float delta_col = (v_col - (kda ? kv_col : g_scalar * kv_col)) * beta_val;

        float attn_partial = 0.0f;
        for (unsigned int r = 0; r < max_rows_per_lane; ++r) {
            if (!kda) {
                g_reg[r] = active_col ? g_scalar : 0.0f;
            }
            s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        reduce[tid] = attn_partial;
        __builtin_amdgcn_s_barrier();

        for (unsigned int step = lanes_per_column / 2; step > 0; step >>= 1) {
            if (lane < step) {
                reduce[tid] += reduce[tid + step];
            }
            __builtin_amdgcn_s_barrier();
        }

        if (active_col && lane == 0) {
            *attn_out = reduce[col_group * lanes_per_column] * c.scale;
        }
        attn_out += c.S_v * c.H;
    }

    for (unsigned int r = 0; r < max_rows_per_lane; ++r) {
        const unsigned int row = r * lanes_per_column + lane;
        if (active_col && row < static_cast<unsigned int>(c.S_v)) {
            state_out[row] = s_shard[r];
        }
    }
}

static __device__ __forceinline__ float hrx_reduce_cluster8(float value) {
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        value += __shfl_down(value, offset, 8);
    }
    return __shfl(value, 0, 8);
}

template <int dpp_ctrl>
static __device__ __forceinline__ float hrx_dpp_shl(float value) {
    return __builtin_bit_cast(
        float,
        __builtin_amdgcn_mov_dpp(
            __builtin_bit_cast(int, value),
            dpp_ctrl,
            0xf,
            0xf,
            true));
}

static __device__ __forceinline__ float hrx_reduce_cluster8_dpp(float value) {
    value += hrx_dpp_shl<0x104>(value); // row_shl:4
    value += hrx_dpp_shl<0x102>(value); // row_shl:2
    value += hrx_dpp_shl<0x101>(value); // row_shl:1
    return __shfl(value, 0, 8);
}

extern "C" __global__ void hrx_gated_delta_net_s128_cluster8_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_f32_constants c) {
    constexpr unsigned int S_v = 128;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int columns_per_workgroup = 8;
    constexpr unsigned int rows_per_lane = S_v / lanes_per_column;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const long long col =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group);
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();

    if (head >= c.H || seq >= c.n_seqs || col_group >= columns_per_workgroup) {
        return;
    }

    const long long iq1 = head % c.neq1;
    const long long ik1 = head % c.nek1;
    const long long iq3 = seq / c.rq3;
    const long long ik3 = seq / c.rk3;
    const bool kda = c.g_ne0 == S_v;
    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * S_v + col;
    float * state_out = state_dst + c.state_dst_offset + (seq * c.H + head) * S_v * S_v + col * S_v;
    const float * state_col = state_in + (seq * c.H + head) * S_v * S_v + col * S_v;

    float s_shard[rows_per_lane];
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = state_col[r * lanes_per_column + lane];
    }

    for (long long token = 0; token < c.n_tokens; ++token) {
        const char * q_base = reinterpret_cast<const char *>(q) + iq3 * c.q_nb3 + token * c.q_nb2 + iq1 * c.q_nb1;
        const char * k_base = reinterpret_cast<const char *>(k) + ik3 * c.k_nb3 + token * c.k_nb2 + ik1 * c.k_nb1;
        const char * v_base = reinterpret_cast<const char *>(v) + seq * c.v_nb3 + token * c.v_nb2 + head * c.v_nb1;
        const char * g_base = reinterpret_cast<const char *>(g) + seq * c.g_nb3 + token * c.g_nb2 + head * c.g_nb1;
        const char * beta_base =
            reinterpret_cast<const char *>(beta) + seq * c.beta_nb3 + token * c.beta_nb2 + head * c.beta_nb1;

        float q_reg[rows_per_lane];
        float k_reg[rows_per_lane];
        float g_reg[rows_per_lane];
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            const unsigned int row = r * lanes_per_column + lane;
            q_reg[r] = *reinterpret_cast<const float *>(q_base + row * sizeof(float));
            k_reg[r] = *reinterpret_cast<const float *>(k_base + row * sizeof(float));
            g_reg[r] = kda ? __builtin_expf(*reinterpret_cast<const float *>(g_base + row * sizeof(float))) : 1.0f;
        }

        float kv_partial = 0.0f;
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            kv_partial += g_reg[r] * s_shard[r] * k_reg[r];
        }
        const float kv_col = hrx_reduce_cluster8(kv_partial);
        const float beta_val = *reinterpret_cast<const float *>(beta_base);
        const float v_col = *reinterpret_cast<const float *>(v_base + col * sizeof(float));
        const float g_scalar = kda ? 1.0f : __builtin_expf(*reinterpret_cast<const float *>(g_base));
        const float delta_col = (v_col - (kda ? kv_col : g_scalar * kv_col)) * beta_val;

        float attn_partial = 0.0f;
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            if (!kda) {
                g_reg[r] = g_scalar;
            }
            s_shard[r] = g_reg[r] * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = hrx_reduce_cluster8(attn_partial);

        if (lane == 0) {
            *attn_out = attn_col * c.scale;
        }
        attn_out += S_v * c.H;
    }

    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        state_out[r * lanes_per_column + lane] = s_shard[r];
    }
}

extern "C" __global__ void hrx_gated_delta_net_s128_cluster8_nokda_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_f32_constants c) {
    constexpr unsigned int S_v = 128;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int columns_per_workgroup = 8;
    constexpr unsigned int rows_per_lane = S_v / lanes_per_column;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const long long col =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group);
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();

    if (head >= c.H || seq >= c.n_seqs || col_group >= columns_per_workgroup) {
        return;
    }

    const long long iq1 = head % c.neq1;
    const long long iq3 = seq / c.rq3;
    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * S_v + col;
    float * state_out = state_dst + c.state_dst_offset + (seq * c.H + head) * S_v * S_v + col * S_v;
    const float * state_col = state_in + (seq * c.H + head) * S_v * S_v + col * S_v;

    float s_shard[rows_per_lane];
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = state_col[r * lanes_per_column + lane];
    }

    for (long long token = 0; token < c.n_tokens; ++token) {
        const char * q_base = reinterpret_cast<const char *>(q) + iq3 * c.q_nb3 + token * c.q_nb2 + iq1 * c.q_nb1;
        const char * k_base = reinterpret_cast<const char *>(k) + iq3 * c.k_nb3 + token * c.k_nb2 + iq1 * c.k_nb1;
        const char * v_base = reinterpret_cast<const char *>(v) + seq * c.v_nb3 + token * c.v_nb2 + head * c.v_nb1;
        const char * g_base = reinterpret_cast<const char *>(g) + seq * c.g_nb3 + token * c.g_nb2 + head * c.g_nb1;
        const char * beta_base =
            reinterpret_cast<const char *>(beta) + seq * c.beta_nb3 + token * c.beta_nb2 + head * c.beta_nb1;

        const float g_scalar = __expf(*reinterpret_cast<const float *>(g_base));
        float k_reg[rows_per_lane];
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            const unsigned int row = r * lanes_per_column + lane;
            k_reg[r] = *reinterpret_cast<const float *>(k_base + row * sizeof(float));
        }

        float kv_partial = 0.0f;
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            kv_partial += g_scalar * s_shard[r] * k_reg[r];
        }
        const float kv_col = hrx_reduce_cluster8(kv_partial);
        const float beta_val = *reinterpret_cast<const float *>(beta_base);
        const float v_col = *reinterpret_cast<const float *>(v_base + col * sizeof(float));
        const float delta_col = (v_col - kv_col) * beta_val;

        float attn_partial = 0.0f;
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            const unsigned int row = r * lanes_per_column + lane;
            const float q_val = *reinterpret_cast<const float *>(q_base + row * sizeof(float));
            s_shard[r] = g_scalar * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_val;
        }
        const float attn_col = hrx_reduce_cluster8(attn_partial);

        if (lane == 0) {
            *attn_out = attn_col * c.scale;
        }
        attn_out += S_v * c.H;
    }

    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        state_out[r * lanes_per_column + lane] = s_shard[r];
    }
}

extern "C" __global__ void hrx_gated_delta_net_s128_cluster8_nokda_nomod_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_s128_nokda_nomod_constants c) {
    constexpr unsigned int S_v = 128;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int columns_per_workgroup = 8;
    constexpr unsigned int rows_per_lane = S_v / lanes_per_column;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const long long col =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group);
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();

    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * S_v + col;
    float * state_out = state_dst + c.state_dst_offset + (seq * c.H + head) * S_v * S_v + col * S_v;
    const float * state_col = state_in + (seq * c.H + head) * S_v * S_v + col * S_v;

    float s_shard[rows_per_lane];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = state_col[r * lanes_per_column + lane];
    }

    for (long long token = 0; token < c.n_tokens; ++token) {
        const long long q_head = head & c.q_head_mask;
        const long long k_head = head & c.k_head_mask;
        const char * q_base = reinterpret_cast<const char *>(q) + seq * c.q_nb3 + token * c.q_nb2 + q_head * c.q_nb1;
        const char * k_base = reinterpret_cast<const char *>(k) + seq * c.k_nb3 + token * c.k_nb2 + k_head * c.k_nb1;
        const char * v_base = reinterpret_cast<const char *>(v) + seq * c.v_nb3 + token * c.v_nb2 + head * c.v_nb1;
        const char * g_base = reinterpret_cast<const char *>(g) + seq * c.g_nb3 + token * c.g_nb2 + head * c.g_nb1;
        const char * beta_base =
            reinterpret_cast<const char *>(beta) + seq * c.beta_nb3 + token * c.beta_nb2 + head * c.beta_nb1;

        const float g_scalar = __expf(*reinterpret_cast<const float *>(g_base));
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        #pragma unroll
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            const unsigned int row = r * lanes_per_column + lane;
            k_reg[r] = *reinterpret_cast<const float *>(k_base + row * sizeof(float));
            q_reg[r] = *reinterpret_cast<const float *>(q_base + row * sizeof(float));
        }

        float kv_partial = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            kv_partial += g_scalar * s_shard[r] * k_reg[r];
        }
        const float kv_col = hrx_reduce_cluster8(kv_partial);
        const float beta_val = *reinterpret_cast<const float *>(beta_base);
        const float v_col = *reinterpret_cast<const float *>(v_base + col * sizeof(float));
        const float delta_col = (v_col - kv_col) * beta_val;

        float attn_partial = 0.0f;
        #pragma unroll
        for (unsigned int r = 0; r < rows_per_lane; ++r) {
            s_shard[r] = g_scalar * s_shard[r] + k_reg[r] * delta_col;
            attn_partial += s_shard[r] * q_reg[r];
        }
        const float attn_col = hrx_reduce_cluster8(attn_partial);

        if (lane == 0) {
            *attn_out = attn_col * c.scale;
        }
        attn_out += S_v * c.H;
    }

    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        state_out[r * lanes_per_column + lane] = s_shard[r];
    }
}

extern "C" __global__ void hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_constants c) {
    constexpr unsigned int S_v = 128;
    constexpr unsigned int H = 32;
    constexpr unsigned int qk_heads = 16;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int columns_per_workgroup = 4;
    constexpr unsigned int rows_per_lane = S_v / lanes_per_column;
    constexpr unsigned int state_head_stride = S_v * S_v;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const unsigned int col = __builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group;
    const unsigned int head = __builtin_amdgcn_workgroup_id_y();
    const unsigned int qk_head = head & (qk_heads - 1);

    const float * q_base = q + qk_head * S_v;
    const float * k_base = k + qk_head * S_v;
    const float * v_base = v + head * S_v;
    const float * state_col = state_in + head * state_head_stride + col * S_v;
    float * state_out = state_dst + c.state_dst_offset + head * state_head_stride + col * S_v;

    float s_shard[rows_per_lane];
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        const unsigned int row = r * lanes_per_column + lane;
        s_shard[r] = state_col[row];
        k_reg[r] = k_base[row];
        q_reg[r] = q_base[row];
    }

    const float g_scalar = __expf(g[head]);

    float kv_partial = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        kv_partial += g_scalar * s_shard[r] * k_reg[r];
    }
    const float kv_col = hrx_reduce_cluster8_dpp(kv_partial);
    const float delta_col = (v_base[col] - kv_col) * beta[head];

    float attn_partial = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = g_scalar * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = hrx_reduce_cluster8_dpp(attn_partial);

    if (lane == 0) {
        dst[head * S_v + col] = attn_col * c.scale;
    }

    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        state_out[r * lanes_per_column + lane] = s_shard[r];
    }
}

extern "C" __global__ void hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_beta_sigmoid_f32(
        const float * q,
        const float * k,
        const float * v,
        const float * g,
        const float * beta,
        const float * state_in,
        float * dst,
        float * state_dst,
        hrx_gated_delta_net_s128_h32_qk16_tok1_nokda_constants c) {
    constexpr unsigned int S_v = 128;
    constexpr unsigned int H = 32;
    constexpr unsigned int qk_heads = 16;
    constexpr unsigned int lanes_per_column = 8;
    constexpr unsigned int columns_per_workgroup = 4;
    constexpr unsigned int rows_per_lane = S_v / lanes_per_column;
    constexpr unsigned int state_head_stride = S_v * S_v;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (lanes_per_column - 1);
    const unsigned int col_group = tid / lanes_per_column;
    const unsigned int col = __builtin_amdgcn_workgroup_id_x() * columns_per_workgroup + col_group;
    const unsigned int head = __builtin_amdgcn_workgroup_id_y();
    const unsigned int qk_head = head & (qk_heads - 1);

    const float * q_base = q + qk_head * S_v;
    const float * k_base = k + qk_head * S_v;
    const float * v_base = v + head * S_v;
    const float * state_col = state_in + head * state_head_stride + col * S_v;
    float * state_out = state_dst + c.state_dst_offset + head * state_head_stride + col * S_v;

    float s_shard[rows_per_lane];
    float k_reg[rows_per_lane];
    float q_reg[rows_per_lane];
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        const unsigned int row = r * lanes_per_column + lane;
        s_shard[r] = state_col[row];
        k_reg[r] = k_base[row];
        q_reg[r] = q_base[row];
    }

    const float g_scalar = __expf(g[head]);

    float kv_partial = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        kv_partial += g_scalar * s_shard[r] * k_reg[r];
    }
    const float kv_col = hrx_reduce_cluster8_dpp(kv_partial);
    const float delta_col = (v_base[col] - kv_col) * hrx_sigmoid_f32(beta[head]);

    float attn_partial = 0.0f;
    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        s_shard[r] = g_scalar * s_shard[r] + k_reg[r] * delta_col;
        attn_partial += s_shard[r] * q_reg[r];
    }
    const float attn_col = hrx_reduce_cluster8_dpp(attn_partial);

    if (lane == 0) {
        dst[head * S_v + col] = attn_col * c.scale;
    }

    #pragma unroll
    for (unsigned int r = 0; r < rows_per_lane; ++r) {
        state_out[r * lanes_per_column + lane] = s_shard[r];
    }
}
