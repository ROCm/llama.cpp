#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q8_0 {
    unsigned short d;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_rhs_q8 {
    unsigned short ds[8];
    int qs[32];
};

struct hrx_q8_0_mmqv_a_cache {
    int qs[8];
    float d;
};

struct hrx_q8_1_mmqv_b_cache_q8 {
    int qs[8];
    float d;
};

static __device__ __forceinline__ int hrx_sdot4_q8_q8_1(int qpack, int rpack) {
    return __builtin_amdgcn_sudot4(true, qpack, true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_q8_0_pack4(const hrx_block_q8_0 * block, int iqs) {
    const uint16_t lo = *reinterpret_cast<const uint16_t *>(block->qs + iqs * 4);
    const uint16_t hi = *reinterpret_cast<const uint16_t *>(block->qs + iqs * 4 + 2);
    return static_cast<int>(static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16));
}

static __device__ __forceinline__ void hrx_q8_0_mmqv_load_a(
        hrx_q8_0_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q8_0 * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q8_0 * block = src0 + row * blocks_per_row + kb;
    buf_a[buf_idx].qs[iqs] = hrx_q8_0_pack4(block, iqs);
    if (iqs == 0) {
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d));
    }
}

static __device__ __forceinline__ void hrx_q8_0_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache_q8 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
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

static __device__ __forceinline__ void hrx_reduce8_256(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = 256 / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
        shared[wave + 4 * waves] = sum4;
        shared[wave + 5 * waves] = sum5;
        shared[wave + 6 * waves] = sum6;
        shared[wave + 7 * waves] = sum7;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    sum4 = lane < waves ? shared[lane + 4 * waves] : 0.0f;
    sum5 = lane < waves ? shared[lane + 5 * waves] : 0.0f;
    sum6 = lane < waves ? shared[lane + 6 * waves] : 0.0f;
    sum7 = lane < waves ? shared[lane + 7 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
            sum4 += __shfl_down(sum4, offset);
            sum5 += __shfl_down(sum5, offset);
            sum6 += __shfl_down(sum6, offset);
            sum7 += __shfl_down(sum7, offset);
        }
    }
}

static __device__ __forceinline__ void hrx_reduce8_128(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = 128 / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
        shared[wave + 4 * waves] = sum4;
        shared[wave + 5 * waves] = sum5;
        shared[wave + 6 * waves] = sum6;
        shared[wave + 7 * waves] = sum7;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    sum4 = lane < waves ? shared[lane + 4 * waves] : 0.0f;
    sum5 = lane < waves ? shared[lane + 5 * waves] : 0.0f;
    sum6 = lane < waves ? shared[lane + 6 * waves] : 0.0f;
    sum7 = lane < waves ? shared[lane + 7 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
            sum4 += __shfl_down(sum4, offset);
            sum5 += __shfl_down(sum5, offset);
            sum6 += __shfl_down(sum6, offset);
            sum7 += __shfl_down(sum7, offset);
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_f32(
        const hrx_block_q8_0 * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const hrx_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = d * static_cast<float>(block->qs[in_block_base + j]);
            sum += value * src1_col[src_base + j];
        }
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_cols8_f32(
        const hrx_block_q8_0 * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const hrx_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col0 + src_base);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col0 + k + src_base);
        const float4 b2 = *reinterpret_cast<const float4 *>(src1_col0 + 2 * k + src_base);
        const float4 b3 = *reinterpret_cast<const float4 *>(src1_col0 + 3 * k + src_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col0 + 4 * k + src_base);
        const float4 b5 = *reinterpret_cast<const float4 *>(src1_col0 + 5 * k + src_base);
        const float4 b6 = *reinterpret_cast<const float4 *>(src1_col0 + 6 * k + src_base);
        const float4 b7 = *reinterpret_cast<const float4 *>(src1_col0 + 7 * k + src_base);
#define HRX_Q8_0_COLS8_ACC(J, FIELD) \
        do { \
            const float value = d * static_cast<float>(block->qs[in_block_base + (J)]); \
            sum0 += value * b0.FIELD; \
            sum1 += value * b1.FIELD; \
            sum2 += value * b2.FIELD; \
            sum3 += value * b3.FIELD; \
            sum4 += value * b4.FIELD; \
            sum5 += value * b5.FIELD; \
            sum6 += value * b6.FIELD; \
            sum7 += value * b7.FIELD; \
        } while (0)
        HRX_Q8_0_COLS8_ACC(0, x);
        HRX_Q8_0_COLS8_ACC(1, y);
        HRX_Q8_0_COLS8_ACC(2, z);
        HRX_Q8_0_COLS8_ACC(3, w);
#undef HRX_Q8_0_COLS8_ACC
    }

    hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
        dst_col0[4 * rows] = sum4;
        dst_col0[5 * rows] = sum5;
        dst_col0[6 * rows] = sum6;
        dst_col0[7 * rows] = sum7;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_add_f32(
        const hrx_block_q8_0 * src0, const float * src1, const float * bias, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const hrx_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float value = d * static_cast<float>(block->qs[in_block_base + j]);
            sum += value * src1_col[src_base + j];
        }
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        const long long out_idx = col * rows + row;
        dst[out_idx] = sum + bias[out_idx];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_add_cols8_f32(
        const hrx_block_q8_0 * src0, const float * src1, const float * bias, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 32) {
        const hrx_block_q8_0 * block = row_blocks + block_idx;
        const float d = __half2float(__ushort_as_half(block->d));
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col0 + src_base);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col0 + k + src_base);
        const float4 b2 = *reinterpret_cast<const float4 *>(src1_col0 + 2 * k + src_base);
        const float4 b3 = *reinterpret_cast<const float4 *>(src1_col0 + 3 * k + src_base);
        const float4 b4 = *reinterpret_cast<const float4 *>(src1_col0 + 4 * k + src_base);
        const float4 b5 = *reinterpret_cast<const float4 *>(src1_col0 + 5 * k + src_base);
        const float4 b6 = *reinterpret_cast<const float4 *>(src1_col0 + 6 * k + src_base);
        const float4 b7 = *reinterpret_cast<const float4 *>(src1_col0 + 7 * k + src_base);
#define HRX_Q8_0_ADD_COLS8_ACC(J, FIELD) \
        do { \
            const float value = d * static_cast<float>(block->qs[in_block_base + (J)]); \
            sum0 += value * b0.FIELD; \
            sum1 += value * b1.FIELD; \
            sum2 += value * b2.FIELD; \
            sum3 += value * b3.FIELD; \
            sum4 += value * b4.FIELD; \
            sum5 += value * b5.FIELD; \
            sum6 += value * b6.FIELD; \
            sum7 += value * b7.FIELD; \
        } while (0)
        HRX_Q8_0_ADD_COLS8_ACC(0, x);
        HRX_Q8_0_ADD_COLS8_ACC(1, y);
        HRX_Q8_0_ADD_COLS8_ACC(2, z);
        HRX_Q8_0_ADD_COLS8_ACC(3, w);
#undef HRX_Q8_0_ADD_COLS8_ACC
    }

    hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        const long long out_idx = col0 * rows + row;
        dst[out_idx] = sum0 + bias[out_idx];
        dst[out_idx + rows] = sum1 + bias[out_idx + rows];
        dst[out_idx + 2 * rows] = sum2 + bias[out_idx + 2 * rows];
        dst[out_idx + 3 * rows] = sum3 + bias[out_idx + 3 * rows];
        dst[out_idx + 4 * rows] = sum4 + bias[out_idx + 4 * rows];
        dst[out_idx + 5 * rows] = sum5 + bias[out_idx + 5 * rows];
        dst[out_idx + 6 * rows] = sum6 + bias[out_idx + 6 * rows];
        dst[out_idx + 7 * rows] = sum7 + bias[out_idx + 7 * rows];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_q8_1_x4_mmq128x32_wg256_f32(
        const hrx_block_q8_0 * src0,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 32;
    constexpr int COLS_PER_THREAD = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int row_lane = static_cast<int>(tid & 127u);
    const int col_lane = static_cast<int>(tid >> 7);
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM + row_lane;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN +
        static_cast<long long>(col_lane * COLS_PER_THREAD);
    if (row >= rows || col_base + COLS_PER_THREAD - 1 >= cols) {
        return;
    }

    __shared__ int b_qs[BN][8];
    __shared__ unsigned short b_d[BN];

    const long long blocks_per_row = k / 32;
    const long long q8_blocks_per_col = k / 32;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;

    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 256) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
            const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
            const int inner = static_cast<int>(linear_block & 3);
            b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
            if (iqs == 0) {
                b_d[c] = rhs->ds[inner * 2 + 0];
            }
        }
        __syncthreads();

        const hrx_block_q8_0 * block = row_blocks + kb;
        const float d = __half2float(__ushort_as_half(block->d));
        int qsum[COLS_PER_THREAD] = {};

        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const int qpack = hrx_q8_0_pack4(block, iqs);
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum[col] += hrx_sdot4_q8_q8_1(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum[col]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        const long long out_idx = (col_base + col) * rows + row;
        dst[out_idx] = sum[col];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_add_q8_1_x4_mmq128x32_wg256_f32(
        const hrx_block_q8_0 * src0,
        const hrx_block_q8_1_x4_rhs_q8 * src1,
        const float * bias,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 32;
    constexpr int COLS_PER_THREAD = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int row_lane = static_cast<int>(tid & 127u);
    const int col_lane = static_cast<int>(tid >> 7);
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM + row_lane;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN +
        static_cast<long long>(col_lane * COLS_PER_THREAD);
    if (row >= rows || col_base + COLS_PER_THREAD - 1 >= cols) {
        return;
    }

    __shared__ int b_qs[BN][8];
    __shared__ unsigned short b_d[BN];

    const long long blocks_per_row = k / 32;
    const long long q8_blocks_per_col = k / 32;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const hrx_block_q8_0 * row_blocks = src0 + row * blocks_per_row;

    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 256) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
            const hrx_block_q8_1_x4_rhs_q8 * rhs = src1 + (linear_block >> 2);
            const int inner = static_cast<int>(linear_block & 3);
            b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
            if (iqs == 0) {
                b_d[c] = rhs->ds[inner * 2 + 0];
            }
        }
        __syncthreads();

        const hrx_block_q8_0 * block = row_blocks + kb;
        const float d = __half2float(__ushort_as_half(block->d));
        int qsum[COLS_PER_THREAD] = {};

        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const int qpack = hrx_q8_0_pack4(block, iqs);
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum[col] += hrx_sdot4_q8_q8_1(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum[col]);
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        const long long out_idx = (col_base + col) * rows + row;
        dst[out_idx] = sum[col] + bias[out_idx];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q8_0_add_rows4_cols4_f32(
        const hrx_block_q8_0 * src0, const float * src1, const float * bias, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = __builtin_amdgcn_workgroup_id_x() * 4;
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= rows || col0 + 3 >= cols) {
        return;
    }

    __shared__ float sumsh0[8 * (128 / 32)];
    __shared__ float sumsh1[8 * (128 / 32)];

    const long long blocks_per_row = k / 32;
    const hrx_block_q8_0 * row_blocks0 = src0 + (row0 + 0) * blocks_per_row;
    const hrx_block_q8_0 * row_blocks1 = src0 + (row0 + 1) * blocks_per_row;
    const hrx_block_q8_0 * row_blocks2 = src0 + (row0 + 2) * blocks_per_row;
    const hrx_block_q8_0 * row_blocks3 = src0 + (row0 + 3) * blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum02 = 0.0f;
    float sum03 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum20 = 0.0f;
    float sum21 = 0.0f;
    float sum22 = 0.0f;
    float sum23 = 0.0f;
    float sum30 = 0.0f;
    float sum31 = 0.0f;
    float sum32 = 0.0f;
    float sum33 = 0.0f;

    const int block_lane = tid & 7;
    const int block_slot = tid >> 3;
    const int in_block_base = block_lane << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 16) {
        const hrx_block_q8_0 * block0 = row_blocks0 + block_idx;
        const hrx_block_q8_0 * block1 = row_blocks1 + block_idx;
        const hrx_block_q8_0 * block2 = row_blocks2 + block_idx;
        const hrx_block_q8_0 * block3 = row_blocks3 + block_idx;
        const float d0 = __half2float(__ushort_as_half(block0->d));
        const float d1 = __half2float(__ushort_as_half(block1->d));
        const float d2 = __half2float(__ushort_as_half(block2->d));
        const float d3 = __half2float(__ushort_as_half(block3->d));
        const long long src_base = block_idx * 32 + in_block_base;
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col0 + src_base);
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col0 + k + src_base);
        const float4 b2 = *reinterpret_cast<const float4 *>(src1_col0 + 2 * k + src_base);
        const float4 b3 = *reinterpret_cast<const float4 *>(src1_col0 + 3 * k + src_base);
#define HRX_Q8_0_ADD_ROWS4_COLS4_ACC(J, FIELD) \
        do { \
            const float value0 = d0 * static_cast<float>(block0->qs[in_block_base + (J)]); \
            const float value1 = d1 * static_cast<float>(block1->qs[in_block_base + (J)]); \
            const float value2 = d2 * static_cast<float>(block2->qs[in_block_base + (J)]); \
            const float value3 = d3 * static_cast<float>(block3->qs[in_block_base + (J)]); \
            sum00 += value0 * b0.FIELD; \
            sum01 += value0 * b1.FIELD; \
            sum02 += value0 * b2.FIELD; \
            sum03 += value0 * b3.FIELD; \
            sum10 += value1 * b0.FIELD; \
            sum11 += value1 * b1.FIELD; \
            sum12 += value1 * b2.FIELD; \
            sum13 += value1 * b3.FIELD; \
            sum20 += value2 * b0.FIELD; \
            sum21 += value2 * b1.FIELD; \
            sum22 += value2 * b2.FIELD; \
            sum23 += value2 * b3.FIELD; \
            sum30 += value3 * b0.FIELD; \
            sum31 += value3 * b1.FIELD; \
            sum32 += value3 * b2.FIELD; \
            sum33 += value3 * b3.FIELD; \
        } while (0)
        HRX_Q8_0_ADD_ROWS4_COLS4_ACC(0, x);
        HRX_Q8_0_ADD_ROWS4_COLS4_ACC(1, y);
        HRX_Q8_0_ADD_ROWS4_COLS4_ACC(2, z);
        HRX_Q8_0_ADD_ROWS4_COLS4_ACC(3, w);
#undef HRX_Q8_0_ADD_ROWS4_COLS4_ACC
    }

    hrx_reduce8_128(sum00, sum01, sum02, sum03, sum10, sum11, sum12, sum13, sumsh0);
    hrx_reduce8_128(sum20, sum21, sum22, sum23, sum30, sum31, sum32, sum33, sumsh1);

    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const long long out_idx = (col0 + c) * rows + row0;
            const float row0_sum = c == 0 ? sum00 : (c == 1 ? sum01 : (c == 2 ? sum02 : sum03));
            const float row1_sum = c == 0 ? sum10 : (c == 1 ? sum11 : (c == 2 ? sum12 : sum13));
            const float row2_sum = c == 0 ? sum20 : (c == 1 ? sum21 : (c == 2 ? sum22 : sum23));
            const float row3_sum = c == 0 ? sum30 : (c == 1 ? sum31 : (c == 2 ? sum32 : sum33));
            dst[out_idx] = row0_sum + bias[out_idx];
            dst[out_idx + 1] = row1_sum + bias[out_idx + 1];
            dst[out_idx + 2] = row2_sum + bias[out_idx + 2];
            dst[out_idx + 3] = row3_sum + bias[out_idx + 3];
        }
    }
}
