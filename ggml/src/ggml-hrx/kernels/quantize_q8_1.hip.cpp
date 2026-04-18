#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q8_1 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_quantize_q8_1_constants {
    long long ne00;
    long long s01;
    long long s02;
    long long s03;
    long long ne0;
    long long ne1;
    long long ne2;
};

extern "C" __global__ void hrx_quantize_q8_1_f32(
        const float * src, hrx_block_q8_1 * dst,
        hrx_quantize_q8_1_constants c) {
    const long long block = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    const long long i1 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const long long z = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());

    const long long i3 = z / c.ne2;
    const long long i2 = z - i3 * c.ne2;
    const long long i0 = block * 32 + lane;
    if (i0 >= c.ne0 || i1 >= c.ne1) {
        return;
    }

    const float value = i0 < c.ne00 ? src[i3 * c.s03 + i2 * c.s02 + i1 * c.s01 + i0] : 0.0f;
    float amax = __builtin_fabsf(value);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor(amax, offset));
    }

    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : static_cast<int>(__builtin_rintf(value / d));
    float sum = static_cast<float>(q);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }

    hrx_block_q8_1 * out = dst + (z * c.ne1 + i1) * (c.ne0 / 32) + block;
    out->qs[lane] = static_cast<int8_t>(q < -128 ? -128 : (q > 127 ? 127 : q));
    if (lane == 0) {
        out->d = __half_as_ushort(__float2half(d));
        out->s = __half_as_ushort(__float2half(sum * d));
    }
}
