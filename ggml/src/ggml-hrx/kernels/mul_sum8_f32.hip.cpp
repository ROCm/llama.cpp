#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_sum8_f32_constants {
    long long rows;
    long long n_tokens;
    long long src0_nb1;
    long long src0_nb2;
    long long scale_nb1;
    long long scale_nb2;
    long long dst_nb1;
};

extern "C" __global__ void hrx_mul_sum8_f32(
        const float * src0,
        const float * scale,
        float * dst,
        hrx_mul_sum8_f32_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long total = c.rows * c.n_tokens;
    if (idx >= total) {
        return;
    }

    const long long row = idx % c.rows;
    const long long token = idx / c.rows;
    const char * src0_base = reinterpret_cast<const char *>(src0) + token * c.src0_nb2 + row * sizeof(float);
    const char * scale_base = reinterpret_cast<const char *>(scale) + token * c.scale_nb2;

    float sum = 0.0f;
#pragma unroll
    for (int id = 0; id < 8; ++id) {
        const float x = *reinterpret_cast<const float *>(src0_base + static_cast<long long>(id) * c.src0_nb1);
        const float s = *reinterpret_cast<const float *>(scale_base + static_cast<long long>(id) * c.scale_nb1);
        sum += x * s;
    }

    *reinterpret_cast<float *>(reinterpret_cast<char *>(dst) + token * c.dst_nb1 + row * sizeof(float)) = sum;
}
