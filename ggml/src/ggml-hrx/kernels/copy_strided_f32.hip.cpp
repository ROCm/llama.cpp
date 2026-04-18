#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_copy_strided_f32_constants {
    long long ncols;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src_nb1;
    long long src_nb2;
    long long src_nb3;
    long long row_size;
};

extern "C" __global__ void hrx_copy_strided_f32(
        const float * src, float * dst,
        hrx_copy_strided_f32_constants c) {
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long row = __builtin_amdgcn_workgroup_id_y();
    if (col >= c.ncols || row >= c.nrows) {
        return;
    }

    const long long i3 = row / (c.ne2 * c.ne1);
    const long long row23 = row - i3 * c.ne2 * c.ne1;
    const long long i2 = row23 / c.ne1;
    const long long i1 = row23 - i2 * c.ne1;
    const char * in = reinterpret_cast<const char *>(src) +
        i1 * c.src_nb1 + i2 * c.src_nb2 + i3 * c.src_nb3;
    char * out = reinterpret_cast<char *>(dst) + row * c.row_size;
    *reinterpret_cast<float *>(out + col * sizeof(float)) =
        *reinterpret_cast<const float *>(in + col * sizeof(float));
}
