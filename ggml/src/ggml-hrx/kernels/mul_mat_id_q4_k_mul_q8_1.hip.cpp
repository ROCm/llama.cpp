#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id_mul {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_block_q8_1_id_mul {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_mul_mat_id_q4_k_mul_q8_1_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long src0_nb1;
    long long src0_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
    long long src1_cols;
    long long scale_nb1;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_id_mul(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ float hrx_reduce_256(float sum, float * shared) {
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

extern "C" __global__ void hrx_mul_mat_id_q4_k_mul_q8_1_f32(
        const hrx_block_q4_K_id_mul * src0,
        const hrx_block_q8_1_id_mul * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_q8_1_constants c) {
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

    __shared__ float sumsh[256];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const long long src1_col_idx = token * c.src1_cols + (c.src1_cols == 1 ? 0 : id_pos);
    const hrx_block_q8_1_id_mul * src1_col = src1 + src1_col_idx * (c.k / 32);
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const hrx_block_q4_K_id_mul * block = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id_mul));
        const hrx_block_q8_1_id_mul * rhs = src1_col + block_idx * 8 + group;

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_id_mul(group, block->scales, &sc, &m);

        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const float d8 = __half2float(__ushort_as_half(rhs->d));
        const float rhs_sum = __half2float(__ushort_as_half(rhs->s));
        const int qs_base = (group >> 1) * 32 + lane;

        int qsum = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t packed = block->qs[qs_base + j];
            const int q = (group & 1) ? (packed >> 4) : (packed & 0x0F);
            qsum += q * static_cast<int>(rhs->qs[lane + j]);
        }
        sum += d * d8 * static_cast<float>(qsum);
        if (lane == 0) {
            sum -= min * rhs_sum;
        }
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum * scale_value;
    }
}
