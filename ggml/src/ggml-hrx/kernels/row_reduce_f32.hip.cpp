#include <hip/hip_runtime.h>

struct hrx_row_reduce_f32_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int pad;
};

struct hrx_l2_norm_pair_f32_constants {
    hrx_row_reduce_f32_constants a;
    hrx_row_reduce_f32_constants b;
};

extern "C" __global__ void hrx_sum_rows_f32(
        const float * src, float * dst, hrx_row_reduce_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[256];
    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const char * src_row = reinterpret_cast<const char *>(src) + i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 256) {
        sum += *reinterpret_cast<const float *>(src_row + col * sizeof(float));
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3) = sumsh[0];
    }
}

extern "C" __global__ void hrx_l2_norm_f32(
        const float * src, float * dst, hrx_row_reduce_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[256];
    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const char * src_row = reinterpret_cast<const char *>(src) + i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) + i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 256) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 128; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    const float denom = __builtin_sqrtf(sumsh[0]);
    const float scale = 1.0f / (denom > c.eps ? denom : c.eps);
    for (long long col = tid; col < c.ncols; col += 256) {
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + col * sizeof(float)) * scale;
    }
}

extern "C" __global__ void hrx_l2_norm_wg128_f32(
        const float * src, float * dst, hrx_row_reduce_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float sumsh[128];
    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const char * src_row = reinterpret_cast<const char *>(src) + i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) + i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 128) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 64; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    const float denom = __builtin_sqrtf(sumsh[0]);
    const float scale = 1.0f / (denom > c.eps ? denom : c.eps);
    for (long long col = tid; col < c.ncols; col += 128) {
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + col * sizeof(float)) * scale;
    }
}

static __device__ __forceinline__ void hrx_l2_norm_row_wg128(
        const float * src, float * dst, hrx_row_reduce_f32_constants c, long long row, unsigned int tid) {
    __shared__ float sumsh[128];
    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
    const char * src_row = reinterpret_cast<const char *>(src) + i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) + i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 128) {
        const float value = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
        sum += value * value;
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 64; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    const float denom = __builtin_sqrtf(sumsh[0]);
    const float scale = 1.0f / (denom > c.eps ? denom : c.eps);
    for (long long col = tid; col < c.ncols; col += 128) {
        *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
            *reinterpret_cast<const float *>(src_row + col * sizeof(float)) * scale;
    }
}

extern "C" __global__ void hrx_l2_norm_pair_wg128_f32(
        const float * src0, float * dst0,
        const float * src1, float * dst1,
        hrx_l2_norm_pair_f32_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int pair = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (pair == 0) {
        if (row >= c.a.nrows) {
            return;
        }
        hrx_l2_norm_row_wg128(src0, dst0, c.a, row, tid);
    } else {
        if (row >= c.b.nrows) {
            return;
        }
        hrx_l2_norm_row_wg128(src1, dst1, c.b, row, tid);
    }
}
