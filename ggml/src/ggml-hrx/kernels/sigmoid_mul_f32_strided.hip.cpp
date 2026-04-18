#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_sigmoid_mul_f32_strided_constants {
    long long ne0;
    long long nrows;
    long long attn_ne0;
    long long attn_ne1;
    long long attn_nb1;
    long long attn_nb2;
    long long gate_ne0;
    long long gate_ne1;
    long long gate_nb1;
    long long gate_nb2;
};

static __device__ __forceinline__ float hrx_sigmoid_value_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void hrx_sigmoid_mul_f32_strided(
        const float * attn,
        const float * gate,
        float * dst,
        hrx_sigmoid_mul_f32_strided_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long n = c.ne0 * c.nrows;
    if (idx >= n) {
        return;
    }

    const long long attn_i0 = idx % c.attn_ne0;
    const long long attn_r0 = idx / c.attn_ne0;
    const long long attn_i1 = attn_r0 % c.attn_ne1;
    const long long attn_i2 = (attn_r0 / c.attn_ne1);

    const long long gate_i0 = idx % c.gate_ne0;
    const long long gate_r0 = idx / c.gate_ne0;
    const long long gate_i1 = gate_r0 % c.gate_ne1;
    const long long gate_i2 = (gate_r0 / c.gate_ne1);

    const char * attn_ptr = reinterpret_cast<const char *>(attn) +
        attn_i0 * sizeof(float) + attn_i1 * c.attn_nb1 + attn_i2 * c.attn_nb2;
    const char * gate_ptr = reinterpret_cast<const char *>(gate) +
        gate_i0 * sizeof(float) + gate_i1 * c.gate_nb1 + gate_i2 * c.gate_nb2;

    const float g = *reinterpret_cast<const float *>(gate_ptr);
    dst[idx] =
        *reinterpret_cast<const float *>(attn_ptr) * hrx_sigmoid_value_f32(g);
}
