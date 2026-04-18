#include <hip/hip_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>

struct hrx_soft_max_f32_constants {
    long long ncols;
    long long nrows;
    long long ne01;
    long long ne02;
    long long mask_nb1;
    long long mask_nb2;
    long long mask_nb3;
    long long mask_ne1;
    long long mask_ne2;
    long long mask_ne3;
    float scale;
    int _pad;
};

static __device__ void hrx_soft_max_f32_row(
        const float * src, const float * mask, float * dst,
        hrx_soft_max_f32_constants c) {
    __shared__ float partial[256];

    const int tid = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    if (row >= c.nrows) {
        return;
    }

    const long long i01 = row % c.ne01;
    const long long i02 = (row / c.ne01) % c.ne02;
    const long long i03 = row / (c.ne01 * c.ne02);
    const float * src_row = src + row * c.ncols;
    float * dst_row = dst + row * c.ncols;
    const float * mask_row = mask ?
        reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(mask) +
            (i01 % c.mask_ne1) * c.mask_nb1 +
            (i02 % c.mask_ne2) * c.mask_nb2 +
            (i03 % c.mask_ne3) * c.mask_nb3) : nullptr;

    float local_max = -FLT_MAX;
    for (long long col = tid; col < c.ncols; col += 256) {
        const float value = src_row[col] * c.scale + (mask_row ? mask_row[col] : 0.0f);
        local_max = fmaxf(local_max, value);
    }

    partial[tid] = local_max;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        }
        __syncthreads();
    }
    const float max_val = partial[0];

    float local_sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 256) {
        const float value =
            expf(src_row[col] * c.scale + (mask_row ? mask_row[col] : 0.0f) - max_val);
        local_sum += value;
    }

    partial[tid] = local_sum;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / partial[0];

    for (long long col = tid; col < c.ncols; col += 256) {
        dst_row[col] =
            expf(src_row[col] * c.scale + (mask_row ? mask_row[col] : 0.0f) - max_val) * inv_sum;
    }
}

extern "C" __global__ void hrx_soft_max_f32(
        const float * src, float * dst,
        hrx_soft_max_f32_constants c) {
    hrx_soft_max_f32_row(src, nullptr, dst, c);
}

extern "C" __global__ void hrx_soft_max_f32_mask(
        const float * src, const float * mask, float * dst,
        hrx_soft_max_f32_constants c) {
    hrx_soft_max_f32_row(src, mask, dst, c);
}
