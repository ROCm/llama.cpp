#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q5_K_q8_1_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct hrx_block_q8_1_rhs_q5 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_rhs_q5 {
    unsigned short ds[8];
    int qs[32];
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_q5_q8_1(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ float hrx_reduce_256_q5_q8_1(float sum, float * shared) {
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

static __device__ __forceinline__ int hrx_sudot4_q5_q8_1(uint32_t qpack, int rpack) {
    return __builtin_amdgcn_sudot4(false, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ uint32_t hrx_q5_k_pack4(
        const hrx_block_q5_K_q8_1_lhs * block,
        int group,
        int iqs) {
    const int qs_base = (group >> 1) * 32 + iqs * 4;
    uint32_t qs = *reinterpret_cast<const uint32_t *>(block->qs + qs_base);
    qs = (qs >> ((group & 1) * 4)) & 0x0F0F0F0Fu;

    const uint32_t qh = *reinterpret_cast<const uint32_t *>(block->qh + iqs * 4);
    return qs | (((qh >> group) & 0x01010101u) << 4);
}

struct hrx_q5_k_mmqv_a_cache {
    int qs[8];
    float d;
    float min;
};

struct hrx_q8_1_mmqv_b_cache {
    int qs[8];
    float d;
    float s;
};

static __device__ __forceinline__ void hrx_q5_k_mmqv_load_a(
        hrx_q5_k_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q5_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q5_K_q8_1_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = static_cast<int>(hrx_q5_k_pack4(block, group, iqs));
    if (iqs == 0) {
        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        buf_a[buf_idx].min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    }
}

static __device__ __forceinline__ void hrx_q5_k_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q5 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
        buf_b[buf_idx].s = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 1]));
    }
}
