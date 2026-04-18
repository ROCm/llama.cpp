#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q8_1 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_packed128 {
    unsigned short ds[8];
    int qs[32];
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

extern "C" __global__ void hrx_quantize_q8_1_x4_f32(
        const float * src, hrx_block_q8_1_x4_packed128 * dst,
        hrx_quantize_q8_1_constants c) {
    const long long block_group = static_cast<long long>(__builtin_amdgcn_workgroup_id_x());
    const long long i1 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const long long z = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    const int tid = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int inner = tid >> 5;
    const int lane = tid & 31;

    const long long i3 = z / c.ne2;
    const long long i2 = z - i3 * c.ne2;
    const long long block = block_group * 4 + inner;
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
    const int q_unclamped = amax == 0.0f ? 0 : static_cast<int>(__builtin_rintf(value / d));
    const int q = q_unclamped < -128 ? -128 : (q_unclamped > 127 ? 127 : q_unclamped);
    float sum = static_cast<float>(q);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor(sum, offset);
    }

    const long long blocks_per_col = c.ne0 / 32;
    const long long linear_block = (z * c.ne1 + i1) * blocks_per_col + block;
    hrx_block_q8_1_x4_packed128 * out = dst + (linear_block >> 2);

    const int wave_lane_base = (tid & (warpSize - 1)) & ~31;
    const unsigned int q1 = static_cast<unsigned char>(
        __builtin_amdgcn_ds_bpermute((wave_lane_base + lane + 1) << 2, q));
    const unsigned int q2 = static_cast<unsigned char>(
        __builtin_amdgcn_ds_bpermute((wave_lane_base + lane + 2) << 2, q));
    const unsigned int q3 = static_cast<unsigned char>(
        __builtin_amdgcn_ds_bpermute((wave_lane_base + lane + 3) << 2, q));
    if ((lane & 3) == 0) {
        const unsigned int q0 = static_cast<unsigned char>(q);
        out->qs[inner * 8 + (lane >> 2)] =
            static_cast<int>(q0 | (q1 << 8) | (q2 << 16) | (q3 << 24));
    }
    if (lane == 0) {
        out->ds[inner * 2 + 0] = __half_as_ushort(__float2half(d));
        out->ds[inner * 2 + 1] = __half_as_ushort(__float2half(sum * d));
    }
}
