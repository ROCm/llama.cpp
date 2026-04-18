#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void hrx_scale_f32(
        const float * src, float * dst,
        long long n, float scale, float bias) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = src[idx] * scale + bias;
    }
}
