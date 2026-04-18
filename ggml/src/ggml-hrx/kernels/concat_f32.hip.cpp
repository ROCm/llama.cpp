#include <hip/hip_runtime.h>

struct hrx_concat_f32_constants {
    long long ne0;
    long long ne1;
    long long src0_ne0;
    long long src0_nb0;
    long long src0_nb1;
    long long src1_nb0;
    long long src1_nb1;
    long long dst_nb0;
    long long dst_nb1;
};

extern "C" __global__ void hrx_concat_f32(
        const float * src0, const float * src1, float * dst,
        hrx_concat_f32_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long n = c.ne0 * c.ne1;
    if (idx >= n) {
        return;
    }

    const long long i0 = idx % c.ne0;
    const long long i1 = idx / c.ne0;
    char * out = reinterpret_cast<char *>(dst) + i0 * c.dst_nb0 + i1 * c.dst_nb1;
    if (i0 < c.src0_ne0) {
        const char * in = reinterpret_cast<const char *>(src0) + i0 * c.src0_nb0 + i1 * c.src0_nb1;
        *reinterpret_cast<float *>(out) = *reinterpret_cast<const float *>(in);
    } else {
        const char * in = reinterpret_cast<const char *>(src1) + (i0 - c.src0_ne0) * c.src1_nb0 + i1 * c.src1_nb1;
        *reinterpret_cast<float *>(out) = *reinterpret_cast<const float *>(in);
    }
}
