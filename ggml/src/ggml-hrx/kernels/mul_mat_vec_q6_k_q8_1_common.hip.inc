#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q6_K_q8_1_lhs {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

struct hrx_block_q8_1_rhs_q6 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_rhs_q6 {
    unsigned short ds[8];
    int qs[32];
};

static __device__ __forceinline__ float hrx_reduce_256_q6_q8_1(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

static __device__ __forceinline__ int hrx_q6_k_value(
        const hrx_block_q6_K_q8_1_lhs * block,
        int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;

    int q = 0;
    if (idx < 32) {
        q = (block->ql[ql_base + lane] & 0x0F) | (((block->qh[qh_base + lane] >> 0) & 3) << 4);
    } else if (idx < 64) {
        q = (block->ql[ql_base + lane + 32] & 0x0F) | (((block->qh[qh_base + lane] >> 2) & 3) << 4);
    } else if (idx < 96) {
        q = (block->ql[ql_base + lane] >> 4) | (((block->qh[qh_base + lane] >> 4) & 3) << 4);
    } else {
        q = (block->ql[ql_base + lane + 32] >> 4) | (((block->qh[qh_base + lane] >> 6) & 3) << 4);
    }
    return q - 32;
}

static __device__ __forceinline__ int hrx_q6_k_scale(
        const hrx_block_q6_K_q8_1_lhs * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1(int q0, int q1, int q2, int q3, const int8_t * rhs) {
    const unsigned int qpack =
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q0))) << 0) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q1))) << 8) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q2))) << 16) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q3))) << 24);
    const int rpack = *reinterpret_cast<const int *>(rhs);
    return __builtin_amdgcn_sudot4(true, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1_packed(int q0, int q1, int q2, int q3, int rpack) {
    const unsigned int qpack =
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q0))) << 0) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q1))) << 8) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q2))) << 16) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q3))) << 24);
    return __builtin_amdgcn_sudot4(true, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1_qpack(int qpack, int rpack) {
    return __builtin_amdgcn_sudot4(true, qpack, true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_q6_k_pack4(
        const hrx_block_q6_K_q8_1_lhs * block,
        int group,
        int iqs) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int lane = iqs * 4;
    const int ql_base = half * 64 + lane + ((group_in_half & 1) ? 32 : 0);
    const int qh_base = half * 32 + lane;
    const int ql_shift = (group_in_half >> 1) * 4;
    const int qh_shift = group_in_half * 2;

    const uint32_t ql = *reinterpret_cast<const uint32_t *>(block->ql + ql_base);
    const uint32_t qh = *reinterpret_cast<const uint32_t *>(block->qh + qh_base);
    const uint32_t qu =
        ((ql >> ql_shift) & 0x0F0F0F0Fu) |
        (((qh >> qh_shift) & 0x03030303u) << 4);
    const uint32_t sign_extend = ((~qu) & 0x20202020u) * 7u;
    return static_cast<int>((qu & 0x1F1F1F1Fu) | sign_extend);
}

struct hrx_q6_k_mmqv_a_cache {
    int qs[8];
    float d[2];
};

struct hrx_q8_1_mmqv_b_cache_q6 {
    int qs[8];
    float d;
};

static __device__ __forceinline__ void hrx_q6_k_mmqv_load_a(
        hrx_q6_k_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q6_K_q8_1_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = hrx_q6_k_pack4(block, group, iqs);
    if (iqs == 0 || iqs == 4) {
        buf_a[buf_idx].d[iqs >> 2] =
            __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, iqs * 4));
    }
}

static __device__ __forceinline__ void hrx_q6_k_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache_q6 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q6 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
    }
}
