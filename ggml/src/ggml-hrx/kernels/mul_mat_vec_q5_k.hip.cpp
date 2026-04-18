#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q5_K {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

static __device__ __forceinline__ void hrx_get_scale_min_k4(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ uint32_t hrx_q5_load_u32_strided16(const uint8_t * base, int offset) {
    return static_cast<uint32_t>(base[offset]) |
        (static_cast<uint32_t>(base[offset + 1]) << 8) |
        (static_cast<uint32_t>(base[offset + 16]) << 16) |
        (static_cast<uint32_t>(base[offset + 17]) << 24);
}

static __device__ __forceinline__ float hrx_q5_k_dot4(
        const hrx_block_q5_K * block, const float * src, int group, int lane,
        uint32_t qs_word, uint32_t qh_word, bool high_nibble) {
    uint8_t sc = 0;
    uint8_t m = 0;
    hrx_get_scale_min_k4(group, block->scales, &sc, &m);

    const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
    const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    const int qh_mask = 1 << group;
    const int nibble_shift = high_nibble ? 4 : 0;

    float sum = 0.0f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int rhs_offset = (j & 1) + ((j >> 1) * 16);
        const uint8_t low = static_cast<uint8_t>((qs_word >> (8 * j + nibble_shift)) & 0x0F);
        const uint8_t high = (static_cast<uint8_t>((qh_word >> (8 * j)) & 0xFF) & qh_mask) ? 16 : 0;
        const float q = static_cast<float>(low + high);
        sum += (d * q - min) * src[group * 32 + lane + rhs_offset];
    }
    return sum;
}

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

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_q5_k_f32_impl(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    const long long blocks_per_row = k / 256;
    const hrx_block_q5_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 15;
    const int block_slot = tid >> 4;
    const int block_stride = WG_SIZE >> 4;
    const int il = block_lane >> 2;
    const int ir = block_lane & 3;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int lane = 4 * ir + 2 * v_in;
    const int group0 = 2 * v_im;
    const int group4 = group0 + 4;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q5_K * block = row_blocks + block_idx;
        const float * src_block = src1_col + block_idx * 256;
        const uint32_t qs0 = hrx_q5_load_u32_strided16(block->qs, (group0 >> 1) * 32 + lane);
        const uint32_t qs4 = hrx_q5_load_u32_strided16(block->qs, (group4 >> 1) * 32 + lane);
        const uint32_t qh = hrx_q5_load_u32_strided16(block->qh, lane);

        sum += hrx_q5_k_dot4(block, src_block, group0,     lane, qs0, qh, false);
        sum += hrx_q5_k_dot4(block, src_block, group0 + 1, lane, qs0, qh, true);
        sum += hrx_q5_k_dot4(block, src_block, group4,     lane, qs4, qh, false);
        sum += hrx_q5_k_dot4(block, src_block, group4 + 1, lane, qs4, qh, true);
    }

    sum = hrx_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_f32(
        const hrx_block_q5_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q5_k_f32_impl<256>(src0, src1, dst, k, rows, cols);
}
