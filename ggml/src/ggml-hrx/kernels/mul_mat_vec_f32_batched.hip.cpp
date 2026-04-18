#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_mat_vec_f32_batched_constants {
    long long k;
    long long rows;
    long long cols;
    long long dst_ne2;
    long long dst_ne3;
    long long src0_ne2;
    long long src0_ne3;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long src1_nb1;
    long long src1_nb2;
    long long src1_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
};

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
    const int waves = 256 / warpSize;

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

static __device__ __forceinline__ void hrx_reduce16_256(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float & sum8,
        float & sum9,
        float & sum10,
        float & sum11,
        float & sum12,
        float & sum13,
        float & sum14,
        float & sum15,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    const int waves = 256 / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
        sum8 += __shfl_down(sum8, offset);
        sum9 += __shfl_down(sum9, offset);
        sum10 += __shfl_down(sum10, offset);
        sum11 += __shfl_down(sum11, offset);
        sum12 += __shfl_down(sum12, offset);
        sum13 += __shfl_down(sum13, offset);
        sum14 += __shfl_down(sum14, offset);
        sum15 += __shfl_down(sum15, offset);
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
        shared[wave + 8 * waves] = sum8;
        shared[wave + 9 * waves] = sum9;
        shared[wave + 10 * waves] = sum10;
        shared[wave + 11 * waves] = sum11;
        shared[wave + 12 * waves] = sum12;
        shared[wave + 13 * waves] = sum13;
        shared[wave + 14 * waves] = sum14;
        shared[wave + 15 * waves] = sum15;
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
    sum8 = lane < waves ? shared[lane + 8 * waves] : 0.0f;
    sum9 = lane < waves ? shared[lane + 9 * waves] : 0.0f;
    sum10 = lane < waves ? shared[lane + 10 * waves] : 0.0f;
    sum11 = lane < waves ? shared[lane + 11 * waves] : 0.0f;
    sum12 = lane < waves ? shared[lane + 12 * waves] : 0.0f;
    sum13 = lane < waves ? shared[lane + 13 * waves] : 0.0f;
    sum14 = lane < waves ? shared[lane + 14 * waves] : 0.0f;
    sum15 = lane < waves ? shared[lane + 15 * waves] : 0.0f;
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
            sum8 += __shfl_down(sum8, offset);
            sum9 += __shfl_down(sum9, offset);
            sum10 += __shfl_down(sum10, offset);
            sum11 += __shfl_down(sum11, offset);
            sum12 += __shfl_down(sum12, offset);
            sum13 += __shfl_down(sum13, offset);
            sum14 += __shfl_down(sum14, offset);
            sum15 += __shfl_down(sum15, offset);
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long i11 = outer % c.cols;
    const long long t = outer / c.cols;
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[256];
    float sum = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = *reinterpret_cast<const float *>(src0_row + i * sizeof(float));
        const float b = *reinterpret_cast<const float *>(src1_col + i * sizeof(float));
        sum += a * b;
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3) =
            sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_cols1_ne2_1_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long i13 = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows || i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i13 * c.src1_nb3;

    __shared__ float sumsh[256];
    float sum = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = *reinterpret_cast<const float *>(src0_row + i * sizeof(float));
        const float b = *reinterpret_cast<const float *>(src1_col + i * sizeof(float));
        sum += a * b;
    }

    sum = hrx_reduce_256(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + row * sizeof(float) + i13 * c.dst_nb3) = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long i13 = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows || i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i03 * c.src0_nb3;
    const char * src1_col = reinterpret_cast<const char *>(src1) + i13 * c.src1_nb3;

    float sum = 0.0f;
#pragma unroll
    for (int iter = 0; iter < 16; ++iter) {
        const unsigned int i = tid * 4 + static_cast<unsigned int>(iter) * 128;
        const float4 a = *reinterpret_cast<const float4 *>(src0_row + i * sizeof(float));
        const float4 b = *reinterpret_cast<const float4 *>(src1_col + i * sizeof(float));
        sum += a.x * b.x;
        sum += a.y * b.y;
        sum += a.z * b.z;
        sum += a.w * b.w;
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }

    if (tid == 0) {
        *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + row * sizeof(float) + i13 * c.dst_nb3) = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_cols8_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long col_group = outer % (c.cols / 8);
    const long long i11 = col_group * 8;
    const long long t = outer / (c.cols / 8);
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[8 * (256 / 32)];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = *reinterpret_cast<const float *>(src0_row + i * sizeof(float));
        const long long rhs = i * sizeof(float);
        sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
        sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
        sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
        sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
        sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
        sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs);
        sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs);
        sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs);
    }

    hrx_reduce8_256(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row) = sum0;
        *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum1;
        *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum2;
        *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum3;
        *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1) = sum4;
        *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1) = sum5;
        *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1) = sum6;
        *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1) = sum7;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_cols16_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long col_group = outer % (c.cols / 16);
    const long long i11 = col_group * 16;
    const long long t = outer / (c.cols / 16);
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row = reinterpret_cast<const char *>(src0) +
        row * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[16 * (256 / 32)];
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    float sum8 = 0.0f;
    float sum9 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum14 = 0.0f;
    float sum15 = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const float a = *reinterpret_cast<const float *>(src0_row + i * sizeof(float));
        const long long rhs = i * sizeof(float);
        sum0 += a * *reinterpret_cast<const float *>(src1_col0 + rhs);
        sum1 += a * *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
        sum2 += a * *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
        sum3 += a * *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
        sum4 += a * *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
        sum5 += a * *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs);
        sum6 += a * *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs);
        sum7 += a * *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs);
        sum8 += a * *reinterpret_cast<const float *>(src1_col0 + 8 * c.src1_nb1 + rhs);
        sum9 += a * *reinterpret_cast<const float *>(src1_col0 + 9 * c.src1_nb1 + rhs);
        sum10 += a * *reinterpret_cast<const float *>(src1_col0 + 10 * c.src1_nb1 + rhs);
        sum11 += a * *reinterpret_cast<const float *>(src1_col0 + 11 * c.src1_nb1 + rhs);
        sum12 += a * *reinterpret_cast<const float *>(src1_col0 + 12 * c.src1_nb1 + rhs);
        sum13 += a * *reinterpret_cast<const float *>(src1_col0 + 13 * c.src1_nb1 + rhs);
        sum14 += a * *reinterpret_cast<const float *>(src1_col0 + 14 * c.src1_nb1 + rhs);
        sum15 += a * *reinterpret_cast<const float *>(src1_col0 + 15 * c.src1_nb1 + rhs);
    }

    hrx_reduce16_256(
        sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7,
        sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh);

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row) = sum0;
        *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum1;
        *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum2;
        *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum3;
        *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1) = sum4;
        *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1) = sum5;
        *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1) = sum6;
        *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1) = sum7;
        *reinterpret_cast<float *>(dst_row + 8 * c.dst_nb1) = sum8;
        *reinterpret_cast<float *>(dst_row + 9 * c.dst_nb1) = sum9;
        *reinterpret_cast<float *>(dst_row + 10 * c.dst_nb1) = sum10;
        *reinterpret_cast<float *>(dst_row + 11 * c.dst_nb1) = sum11;
        *reinterpret_cast<float *>(dst_row + 12 * c.dst_nb1) = sum12;
        *reinterpret_cast<float *>(dst_row + 13 * c.dst_nb1) = sum13;
        *reinterpret_cast<float *>(dst_row + 14 * c.dst_nb1) = sum14;
        *reinterpret_cast<float *>(dst_row + 15 * c.dst_nb1) = sum15;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_f32_batched_rows2_cols8_f32(
        const float * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f32_batched_constants c) {
    const long long row0 = __builtin_amdgcn_workgroup_id_x() * 2;
    const long long row1 = row0 + 1;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row1 >= c.rows) {
        return;
    }

    const long long col_group = outer % (c.cols / 8);
    const long long i11 = col_group * 8;
    const long long t = outer / (c.cols / 8);
    const long long i12 = t % c.dst_ne2;
    const long long i13 = t / c.dst_ne2;
    if (i13 >= c.dst_ne3) {
        return;
    }

    const long long src0_i02 = c.src0_ne2 == c.dst_ne2 ? i12 : i12 / (c.dst_ne2 / c.src0_ne2);
    const long long src0_i03 = c.src0_ne3 == c.dst_ne3 ? i13 : i13 / (c.dst_ne3 / c.src0_ne3);
    const char * src0_row0 = reinterpret_cast<const char *>(src0) +
        row0 * c.src0_nb1 + src0_i02 * c.src0_nb2 + src0_i03 * c.src0_nb3;
    const char * src0_row1 = src0_row0 + c.src0_nb1;
    const char * src1_col0 = reinterpret_cast<const char *>(src1) +
        i11 * c.src1_nb1 + i12 * c.src1_nb2 + i13 * c.src1_nb3;

    __shared__ float sumsh[16 * (256 / 32)];
    float sum00 = 0.0f;
    float sum01 = 0.0f;
    float sum02 = 0.0f;
    float sum03 = 0.0f;
    float sum04 = 0.0f;
    float sum05 = 0.0f;
    float sum06 = 0.0f;
    float sum07 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum14 = 0.0f;
    float sum15 = 0.0f;
    float sum16 = 0.0f;
    float sum17 = 0.0f;
    for (long long i = tid; i < c.k; i += 256) {
        const long long rhs = i * sizeof(float);
        const float a0 = *reinterpret_cast<const float *>(src0_row0 + rhs);
        const float a1 = *reinterpret_cast<const float *>(src0_row1 + rhs);
        const float b0 = *reinterpret_cast<const float *>(src1_col0 + rhs);
        const float b1 = *reinterpret_cast<const float *>(src1_col0 + c.src1_nb1 + rhs);
        const float b2 = *reinterpret_cast<const float *>(src1_col0 + 2 * c.src1_nb1 + rhs);
        const float b3 = *reinterpret_cast<const float *>(src1_col0 + 3 * c.src1_nb1 + rhs);
        const float b4 = *reinterpret_cast<const float *>(src1_col0 + 4 * c.src1_nb1 + rhs);
        const float b5 = *reinterpret_cast<const float *>(src1_col0 + 5 * c.src1_nb1 + rhs);
        const float b6 = *reinterpret_cast<const float *>(src1_col0 + 6 * c.src1_nb1 + rhs);
        const float b7 = *reinterpret_cast<const float *>(src1_col0 + 7 * c.src1_nb1 + rhs);
        sum00 += a0 * b0;
        sum01 += a0 * b1;
        sum02 += a0 * b2;
        sum03 += a0 * b3;
        sum04 += a0 * b4;
        sum05 += a0 * b5;
        sum06 += a0 * b6;
        sum07 += a0 * b7;
        sum10 += a1 * b0;
        sum11 += a1 * b1;
        sum12 += a1 * b2;
        sum13 += a1 * b3;
        sum14 += a1 * b4;
        sum15 += a1 * b5;
        sum16 += a1 * b6;
        sum17 += a1 * b7;
    }

    hrx_reduce16_256(
        sum00, sum01, sum02, sum03, sum04, sum05, sum06, sum07,
        sum10, sum11, sum12, sum13, sum14, sum15, sum16, sum17, sumsh);

    if (tid == 0) {
        char * dst_row = reinterpret_cast<char *>(dst) +
            row0 * sizeof(float) + i11 * c.dst_nb1 + i12 * c.dst_nb2 + i13 * c.dst_nb3;
        *reinterpret_cast<float *>(dst_row) = sum00;
        *reinterpret_cast<float *>(dst_row + sizeof(float)) = sum10;
        *reinterpret_cast<float *>(dst_row + c.dst_nb1) = sum01;
        *reinterpret_cast<float *>(dst_row + c.dst_nb1 + sizeof(float)) = sum11;
        *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1) = sum02;
        *reinterpret_cast<float *>(dst_row + 2 * c.dst_nb1 + sizeof(float)) = sum12;
        *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1) = sum03;
        *reinterpret_cast<float *>(dst_row + 3 * c.dst_nb1 + sizeof(float)) = sum13;
        *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1) = sum04;
        *reinterpret_cast<float *>(dst_row + 4 * c.dst_nb1 + sizeof(float)) = sum14;
        *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1) = sum05;
        *reinterpret_cast<float *>(dst_row + 5 * c.dst_nb1 + sizeof(float)) = sum15;
        *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1) = sum06;
        *reinterpret_cast<float *>(dst_row + 6 * c.dst_nb1 + sizeof(float)) = sum16;
        *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1) = sum07;
        *reinterpret_cast<float *>(dst_row + 7 * c.dst_nb1 + sizeof(float)) = sum17;
    }
}
