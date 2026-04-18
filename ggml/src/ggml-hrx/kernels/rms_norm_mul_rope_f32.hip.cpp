#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

struct hrx_rms_norm_mul_rope_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long weight_ne0;
    long long weight_ne1;
    long long weight_ne2;
    long long weight_ne3;
    long long weight_nb1;
    long long weight_nb2;
    long long weight_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int _pad0;
    int n_dims;
    int mode;
    int section0;
    int section1;
    int section2;
    int section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad1;
};

struct hrx_rms_norm_mul_rope_set_rows_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long weight_ne0;
    long long weight_ne1;
    long long weight_ne2;
    long long weight_ne3;
    long long weight_nb1;
    long long weight_nb2;
    long long weight_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    float eps;
    int _pad0;
    int n_dims;
    int mode;
    int section0;
    int section1;
    int section2;
    int section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad1;
    long long set_rows_ne1;
    int set_rows_ne11;
    int set_rows_ne12;
    long long idx_nb0;
    long long idx_nb1;
    long long idx_nb2;
    long long set_rows_dst_nb1;
    long long set_rows_dst_nb2;
    long long set_rows_dst_nb3;
};

static __device__ long long hrx_load_i64(const long long * base, long long byte_offset) {
    return *reinterpret_cast<const long long *>(reinterpret_cast<const char *>(base) + byte_offset);
}

template <typename C>
static __device__ void hrx_row_coords(long long row, const C & c, long long & i1, long long & i2, long long & i3) {
    i3 = row / (c.ne1 * c.ne2);
    i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;
}

template <typename C>
static __device__ int hrx_rope_pos_idx(int i0, const C & c) {
    const int sect_dims = c.section0 + c.section1 + c.section2 + c.section3;
    const int sector = (i0 / 2) % sect_dims;
    return
        (sector % 3 == 1 && sector < 3 * c.section1) ? 1 :
        (sector % 3 == 2 && sector < 3 * c.section2) ? 2 :
        (sector % 3 == 0 && sector < 3 * c.section0) ? 0 : 3;
}

template <typename C>
static __device__ void hrx_store_rope_pair_f32(
        const float * normed, const int * pos, float * dst, const C & c,
        long long i1, long long i2, long long i3, int i0) {
    char * dst_row = reinterpret_cast<char *>(dst) + i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;
    if (i0 >= c.n_dims) {
        *reinterpret_cast<float *>(dst_row + i0 * static_cast<long long>(sizeof(float))) = normed[i0 + 0];
        *reinterpret_cast<float *>(dst_row + (i0 + 1) * static_cast<long long>(sizeof(float))) = normed[i0 + 1];
        return;
    }

    const int pos_idx = hrx_rope_pos_idx(i0, c);
    const float theta_scale = powf(c.freq_base, -2.0f / static_cast<float>(c.n_dims));
    const float theta = static_cast<float>(pos[i2 + c.ne2 * pos_idx]) *
        powf(theta_scale, static_cast<float>(i0) / 2.0f) * c.freq_scale;
    const float cos_theta = cosf(theta) * c.attn_factor;
    const float sin_theta = sinf(theta) * c.attn_factor;

    const long long off0 = i0 / 2;
    const long long off1 = off0 + c.n_dims / 2;
    const float x0 = normed[off0];
    const float x1 = normed[off1];
    *reinterpret_cast<float *>(dst_row + off0 * static_cast<long long>(sizeof(float))) = x0 * cos_theta - x1 * sin_theta;
    *reinterpret_cast<float *>(dst_row + off1 * static_cast<long long>(sizeof(float))) = x0 * sin_theta + x1 * cos_theta;
}

static __device__ void hrx_store_rope_pair_f16_set_rows(
        const float * normed, const int * pos, const long long * idxs, __half * dst,
        const hrx_rms_norm_mul_rope_set_rows_constants & c,
        long long i1, long long i2, long long i3, int i0) {
    const long long i12 = i3 % c.set_rows_ne12;
    const long long i11 = i2 % c.set_rows_ne11;
    const long long dst_row = hrx_load_i64(idxs, i2 * c.idx_nb0 + i11 * c.idx_nb1 + i12 * c.idx_nb2);
    if (dst_row < 0 || dst_row >= c.set_rows_ne1) {
        return;
    }

    const long long flat_base_bytes =
        i1 * c.dst_nb1 +
        dst_row * c.set_rows_dst_nb1 +
        i3 * c.set_rows_dst_nb2;

    if (i0 >= c.n_dims) {
        *reinterpret_cast<__half *>(reinterpret_cast<char *>(dst) + flat_base_bytes + i0 * static_cast<long long>(sizeof(__half))) =
            __float2half(normed[i0 + 0]);
        *reinterpret_cast<__half *>(reinterpret_cast<char *>(dst) + flat_base_bytes + (i0 + 1) * static_cast<long long>(sizeof(__half))) =
            __float2half(normed[i0 + 1]);
        return;
    }

    const int pos_idx = hrx_rope_pos_idx(i0, c);
    const float theta_scale = powf(c.freq_base, -2.0f / static_cast<float>(c.n_dims));
    const float theta = static_cast<float>(pos[i2 + c.ne2 * pos_idx]) *
        powf(theta_scale, static_cast<float>(i0) / 2.0f) * c.freq_scale;
    const float cos_theta = cosf(theta) * c.attn_factor;
    const float sin_theta = sinf(theta) * c.attn_factor;

    const long long off0 = i0 / 2;
    const long long off1 = off0 + c.n_dims / 2;
    const float x0 = normed[off0];
    const float x1 = normed[off1];
    *reinterpret_cast<__half *>(reinterpret_cast<char *>(dst) + flat_base_bytes + off0 * static_cast<long long>(sizeof(__half))) =
        __float2half(x0 * cos_theta - x1 * sin_theta);
    *reinterpret_cast<__half *>(reinterpret_cast<char *>(dst) + flat_base_bytes + off1 * static_cast<long long>(sizeof(__half))) =
        __float2half(x0 * sin_theta + x1 * cos_theta);
}

template <typename C>
static __device__ void hrx_compute_normed(
        const float * src, const float * weight, float * normed,
        const C & c, long long i1, long long i2, long long i3, unsigned int tid) {
    __shared__ float sumsh[512];

    const long long wi1 = c.weight_ne1 == 1 ? 0 : i1;
    const long long wi2 = c.weight_ne2 == 1 ? 0 : i2;
    const long long wi3 = c.weight_ne3 == 1 ? 0 : i3;
    const char * src_row = reinterpret_cast<const char *>(src) + i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    const char * weight_row = reinterpret_cast<const char *>(weight) + wi1 * c.weight_nb1 + wi2 * c.weight_nb2 + wi3 * c.weight_nb3;

    float sum = 0.0f;
    for (long long col = tid; col < c.ncols; col += 512) {
        const float value = *reinterpret_cast<const float *>(src_row + col * static_cast<long long>(sizeof(float)));
        sum += value * value;
    }

    sumsh[tid] = sum;
    __builtin_amdgcn_s_barrier();

    for (unsigned int step = 256; step > 0; step >>= 1) {
        if (tid < step) {
            sum += sumsh[tid + step];
            sumsh[tid] = sum;
        }
        __builtin_amdgcn_s_barrier();
    }

    const float scale = 1.0f / __builtin_sqrtf(sumsh[0] / static_cast<float>(c.ncols) + c.eps);
    for (long long col = tid; col < c.ncols; col += 512) {
        const long long wcol = c.weight_ne0 == 1 ? 0 : col;
        const float src_value = *reinterpret_cast<const float *>(src_row + col * static_cast<long long>(sizeof(float)));
        const float weight_value = *reinterpret_cast<const float *>(weight_row + wcol * static_cast<long long>(sizeof(float)));
        normed[col] = src_value * scale * weight_value;
    }
    __builtin_amdgcn_s_barrier();
}

extern "C" __global__ void hrx_rms_norm_mul_rope_f32(
        const float * src, const float * weight, const int * pos, float * dst,
        hrx_rms_norm_mul_rope_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float normed[1024];
    long long i1 = 0;
    long long i2 = 0;
    long long i3 = 0;
    hrx_row_coords(row, c, i1, i2, i3);
    hrx_compute_normed(src, weight, normed, c, i1, i2, i3, tid);

    for (int i0 = static_cast<int>(2 * tid); i0 < c.ncols; i0 += 1024) {
        hrx_store_rope_pair_f32(normed, pos, dst, c, i1, i2, i3, i0);
    }
}

extern "C" __global__ void hrx_rms_norm_mul_rope_set_rows_f32_f16(
        const float * src, const float * weight, const int * pos, const long long * idxs, __half * dst,
        hrx_rms_norm_mul_rope_set_rows_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.nrows) {
        return;
    }

    __shared__ float normed[1024];
    long long i1 = 0;
    long long i2 = 0;
    long long i3 = 0;
    hrx_row_coords(row, c, i1, i2, i3);
    hrx_compute_normed(src, weight, normed, c, i1, i2, i3, tid);

    for (int i0 = static_cast<int>(2 * tid); i0 < c.ncols; i0 += 1024) {
        hrx_store_rope_pair_f16_set_rows(normed, pos, idxs, dst, c, i1, i2, i3, i0);
    }
}
