#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_copy_f32_f16_constants {
    long long n;
};

extern "C" __global__ void hrx_copy_f32_f16(
        const float * src, __half * dst,
        hrx_copy_f32_f16_constants c) {
    const long long i = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (i >= c.n) {
        return;
    }

    dst[i] = __float2half(src[i]);
}
