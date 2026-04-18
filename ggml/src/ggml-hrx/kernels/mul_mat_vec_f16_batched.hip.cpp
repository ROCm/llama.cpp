#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_mat_vec_f16_batched_constants {
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

extern "C" __global__ void hrx_mul_mat_vec_f16_batched_f32(
        const __half * src0, const float * src1, float * dst,
        hrx_mul_mat_vec_f16_batched_constants c) {
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
        const float a = __half2float(*reinterpret_cast<const __half *>(src0_row + i * sizeof(__half)));
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
