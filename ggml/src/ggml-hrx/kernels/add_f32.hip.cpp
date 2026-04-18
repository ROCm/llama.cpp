#include <hip/hip_runtime.h>

extern "C" __global__ void hrx_add_f32(
        const float * src0, const float * src1, float * dst,
        long long n) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (idx < n) {
        dst[idx] = src0[idx] + src1[idx];
    }
}
