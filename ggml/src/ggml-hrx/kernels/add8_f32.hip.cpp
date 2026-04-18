#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_add8_f32_constants {
    long long ncols;
    long long nrows;
    long long src0_nb1;
    long long src1_nb1;
    long long src2_nb1;
    long long src3_nb1;
    long long src4_nb1;
    long long src5_nb1;
    long long src6_nb1;
    long long src7_nb1;
    long long dst_nb1;
};

extern "C" __global__ void hrx_add8_f32(
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        const float * src4,
        const float * src5,
        const float * src6,
        const float * src7,
        float * dst,
        hrx_add8_f32_constants c) {
    const long long idx = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long total = c.ncols * c.nrows;
    if (idx >= total) {
        return;
    }

    const long long col = idx % c.ncols;
    const long long row = idx / c.ncols;

    const char * src0_row = reinterpret_cast<const char *>(src0) + row * c.src0_nb1;
    const char * src1_row = reinterpret_cast<const char *>(src1) + row * c.src1_nb1;
    const char * src2_row = reinterpret_cast<const char *>(src2) + row * c.src2_nb1;
    const char * src3_row = reinterpret_cast<const char *>(src3) + row * c.src3_nb1;
    const char * src4_row = reinterpret_cast<const char *>(src4) + row * c.src4_nb1;
    const char * src5_row = reinterpret_cast<const char *>(src5) + row * c.src5_nb1;
    const char * src6_row = reinterpret_cast<const char *>(src6) + row * c.src6_nb1;
    const char * src7_row = reinterpret_cast<const char *>(src7) + row * c.src7_nb1;
    char * dst_row = reinterpret_cast<char *>(dst) + row * c.dst_nb1;
    const long long off = col * static_cast<long long>(sizeof(float));

    *reinterpret_cast<float *>(dst_row + off) =
        *reinterpret_cast<const float *>(src0_row + off) +
        *reinterpret_cast<const float *>(src1_row + off) +
        *reinterpret_cast<const float *>(src2_row + off) +
        *reinterpret_cast<const float *>(src3_row + off) +
        *reinterpret_cast<const float *>(src4_row + off) +
        *reinterpret_cast<const float *>(src5_row + off) +
        *reinterpret_cast<const float *>(src6_row + off) +
        *reinterpret_cast<const float *>(src7_row + off);
}
