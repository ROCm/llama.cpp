#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

struct hrx_rope_f32_constants {
    long long ne00;
    long long ne01;
    long long ne02;
    long long nrows;
    long long src_s1;
    long long src_s2;
    long long src_s3;
    long long dst_s1;
    long long dst_s2;
    long long dst_s3;
    int n_dims;
    int mode;
    int section0;
    int section1;
    int section2;
    int section3;
    float freq_base;
    float freq_scale;
    float attn_factor;
    float _pad;
};

extern "C" __global__ void hrx_rope_f32(
        const float * src, const int * pos, float * dst,
        hrx_rope_f32_constants c) {
    const long long pair = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long pairs_per_row = c.ne00 / 2;
    const long long total_pairs = c.nrows * pairs_per_row;
    if (pair >= total_pairs) {
        return;
    }

    const long long row = pair / pairs_per_row;
    const long long pair_in_row = pair - row * pairs_per_row;
    const int i0 = static_cast<int>(2 * pair_in_row);
    const long long i3 = row / (c.ne01 * c.ne02);
    const long long i2 = (row - i3 * c.ne01 * c.ne02) / c.ne01;
    const long long i1 = row - i3 * c.ne01 * c.ne02 - i2 * c.ne01;
    const long long src_base = i1 * c.src_s1 + i2 * c.src_s2 + i3 * c.src_s3;
    const long long dst_base = i1 * c.dst_s1 + i2 * c.dst_s2 + i3 * c.dst_s3;

    if (i0 >= c.n_dims) {
        dst[dst_base + i0 + 0] = src[src_base + i0 + 0];
        dst[dst_base + i0 + 1] = src[src_base + i0 + 1];
        return;
    }

    const int sect_dims = c.section0 + c.section1 + c.section2 + c.section3;
    const int sector = (i0 / 2) % sect_dims;
    const int pos_idx =
        (sector % 3 == 1 && sector < 3 * c.section1) ? 1 :
        (sector % 3 == 2 && sector < 3 * c.section2) ? 2 :
        (sector % 3 == 0 && sector < 3 * c.section0) ? 0 : 3;

    const float theta_scale = powf(c.freq_base, -2.0f / static_cast<float>(c.n_dims));
    const float theta = static_cast<float>(pos[i2 + c.ne02 * pos_idx]) *
        powf(theta_scale, static_cast<float>(i0) / 2.0f) * c.freq_scale;
    const float cos_theta = cosf(theta) * c.attn_factor;
    const float sin_theta = sinf(theta) * c.attn_factor;

    const long long off0 = i0 / 2;
    const long long off1 = off0 + c.n_dims / 2;
    const float x0 = src[src_base + off0];
    const float x1 = src[src_base + off1];
    dst[dst_base + off0] = x0 * cos_theta - x1 * sin_theta;
    dst[dst_base + off1] = x0 * sin_theta + x1 * cos_theta;
}
