#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_mul_mat_id_q4_k_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_id(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
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
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_f32_impl(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[(WG_SIZE + 31) / 32];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q4_K_id * block = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id));

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_id(group, block->scales, &sc, &m);

        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t packed = block->qs[qs_base + j];
            const float q = (group & 1) ?
                static_cast<float>(packed >> 4) :
                static_cast<float>(packed & 0x0F);
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            sum += (d * q - min) * b;
        }
    }

    sum = hrx_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    hrx_mul_mat_id_q4_k_f32_impl<256>(src0, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_wg64_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    hrx_mul_mat_id_q4_k_f32_impl<64>(src0, src1, ids, dst, c);
}
