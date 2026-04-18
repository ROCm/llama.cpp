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

    float * attn_out = dst + (seq * c.n_tokens * c.H + head) * c.S_v + safe_col;
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
