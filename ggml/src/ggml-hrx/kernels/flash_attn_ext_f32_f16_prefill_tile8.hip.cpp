#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct hrx_flash_attn_ext_f32_f16_prefill_tile8_constants {
    long long D;
    long long KV;
    long long N;
    long long H;
    long long H_KV;
    long long S;
    long long q_nb1;
    long long q_nb2;
    long long q_nb3;
    long long k_nb1;
    long long k_nb2;
    long long k_nb3;
    long long v_nb1;
    long long v_nb2;
    long long v_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    long long mask_nb0;
    long long mask_nb1;
    long long mask_nb3;
    float scale;
    int has_mask;
    float max_bias;
    float m0;
    float m1;
    float logit_softcap;
    int n_head_log2;
    int has_sinks;
};

static __device__ __forceinline__ float hrx_load_f16_tile8(const __half * base, long long byte_offset) {
    return __half2float(*reinterpret_cast<const __half *>(reinterpret_cast<const char *>(base) + byte_offset));
}

static __device__ __forceinline__ float hrx_alibi_slope_tile8(
        const hrx_flash_attn_ext_f32_f16_prefill_tile8_constants c,
        long long head) {
    if (c.max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = head < c.n_head_log2 ? c.m0 : c.m1;
    const int exp_h = head < c.n_head_log2 ? static_cast<int>(head + 1) :
        static_cast<int>(2 * (head - c.n_head_log2) + 1);
    return powf(base, exp_h);
}

extern "C" __global__ void hrx_flash_attn_ext_f32_f16_prefill_tile8(
        const float * q,
        const __half * k,
        const __half * v,
        const __half * mask,
        const float * sinks,
        float * dst,
        hrx_flash_attn_ext_f32_f16_prefill_tile8_constants c) {
    constexpr int BR = 8;
    constexpr int WG = 256;
    __shared__ float logits[BR][512];
    __shared__ float partial[BR][WG];

    const long long tile = __builtin_amdgcn_workgroup_id_x();
    const long long head = __builtin_amdgcn_workgroup_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long token_base = tile * BR;

    if (head >= c.H || seq >= c.S || c.D != 256 || c.KV != 512) {
        return;
    }

    const long long kv_group = c.H / c.H_KV;
    const long long kv_head = head / kv_group;
    const char * k_head = reinterpret_cast<const char *>(k) + kv_head * c.k_nb2 + seq * c.k_nb3;
    const char * v_head = reinterpret_cast<const char *>(v) + kv_head * c.v_nb2 + seq * c.v_nb3;
    const float slope = hrx_alibi_slope_tile8(c, head);
    const float sink = c.has_sinks ? sinks[head] : -FLT_MAX;

    float local_max[BR];
#pragma unroll
    for (int r = 0; r < BR; ++r) {
        local_max[r] = sink;
    }

    for (long long idx = tid; idx < BR * c.KV; idx += WG) {
        const int r = static_cast<int>(idx / c.KV);
        const long long t = idx - static_cast<long long>(r) * c.KV;
        const long long token = token_base + r;
        if (token >= c.N) {
            continue;
        }

        const char * mask_row = reinterpret_cast<const char *>(mask) + token * c.mask_nb1 + seq * c.mask_nb3;
        float mask_value = 0.0f;
        if (c.has_mask) {
            mask_value = hrx_load_f16_tile8(reinterpret_cast<const __half *>(mask_row), t * c.mask_nb0);
            if (mask_value <= -60000.0f) {
                logits[r][t] = -FLT_MAX;
                continue;
            }
        }

        const char * q_row = reinterpret_cast<const char *>(q) + token * c.q_nb1 + head * c.q_nb2 + seq * c.q_nb3;
        const char * k_row = k_head + t * c.k_nb1;
        float score = 0.0f;
#pragma unroll
        for (long long d = 0; d < 256; ++d) {
            const float qv = *reinterpret_cast<const float *>(q_row + d * static_cast<long long>(sizeof(float)));
            score += qv * hrx_load_f16_tile8(reinterpret_cast<const __half *>(k_row), d * static_cast<long long>(sizeof(__half)));
        }
        score *= c.scale;
        if (c.logit_softcap != 0.0f) {
            score = c.logit_softcap * tanhf(score);
        }
        if (c.has_mask) {
            score += slope * mask_value;
        }
        logits[r][t] = score;
        local_max[r] = fmaxf(local_max[r], score);
    }

#pragma unroll
    for (int r = 0; r < BR; ++r) {
        partial[r][tid] = local_max[r];
    }
    __syncthreads();

    for (int stride = WG >> 1; stride > 0; stride >>= 1) {
        if (tid < static_cast<unsigned int>(stride)) {
#pragma unroll
            for (int r = 0; r < BR; ++r) {
                partial[r][tid] = fmaxf(partial[r][tid], partial[r][tid + stride]);
            }
        }
        __syncthreads();
    }

    float local_sum[BR];
#pragma unroll
    for (int r = 0; r < BR; ++r) {
        local_sum[r] = (c.has_sinks && tid == 0 && token_base + r < c.N) ? expf(sink - partial[r][0]) : 0.0f;
    }

    for (long long idx = tid; idx < BR * c.KV; idx += WG) {
        const int r = static_cast<int>(idx / c.KV);
        const long long t = idx - static_cast<long long>(r) * c.KV;
        if (token_base + r >= c.N) {
            continue;
        }
        const float prob = expf(logits[r][t] - partial[r][0]);
        logits[r][t] = prob;
        local_sum[r] += prob;
    }

#pragma unroll
    for (int r = 0; r < BR; ++r) {
        partial[r][tid] = local_sum[r];
    }
    __syncthreads();

    for (int stride = WG >> 1; stride > 0; stride >>= 1) {
        if (tid < static_cast<unsigned int>(stride)) {
#pragma unroll
            for (int r = 0; r < BR; ++r) {
                partial[r][tid] += partial[r][tid + stride];
            }
        }
        __syncthreads();
    }

    if (tid < 256) {
        for (int r = 0; r < BR; ++r) {
            const long long token = token_base + r;
            if (token >= c.N) {
                continue;
            }
            const float inv_sum = 1.0f / partial[r][0];
            float local = 0.0f;
#pragma unroll 4
            for (long long t = 0; t < 512; ++t) {
                const char * v_row = v_head + t * c.v_nb1;
                local += logits[r][t] * inv_sum *
                    hrx_load_f16_tile8(reinterpret_cast<const __half *>(v_row), tid * static_cast<long long>(sizeof(__half)));
            }
            char * dst_row = reinterpret_cast<char *>(dst) + head * c.dst_nb1 + token * c.dst_nb2 + seq * c.dst_nb3;
            *reinterpret_cast<float *>(dst_row + tid * static_cast<long long>(sizeof(float))) = local;
        }
    }
}
