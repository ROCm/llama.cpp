#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_wg(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < ((WG_SIZE + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

static __device__ __forceinline__ int hrx_q6_k_scale(
        const hrx_block_q6_K * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ float hrx_q6_k_dot4(
        const hrx_block_q6_K * block,
        const float * src,
        float d,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);
    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const int q = (ql | (qh << 4)) - 32;
        sum += d * static_cast<float>(q) * src[j];
    }

    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_q6_k_f32_impl(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, lane));

        sum += hrx_q6_k_dot4(block, src1_col + src_base, d, group, lane);
    }

    sum = hrx_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_f32_impl<256>(src0, src1, dst, k, rows, cols);
}
