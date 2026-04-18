#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_get_rows_f32_constants {
    long long nc;
    long long nr;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long idx_nb0;
    long long idx_nb1;
    long long idx_nb2;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
    long long ne10;
    long long ne11;
};

extern "C" __global__ void hrx_get_rows_f32(
        const float * src0, const int * idx, float * dst,
        hrx_get_rows_f32_constants c) {
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long row = __builtin_amdgcn_workgroup_id_y();
    if (col >= c.nc || row >= c.nr) {
        return;
    }

    const long long i12 = row / (c.ne11 * c.ne10);
    const long long i11 = (row - i12 * c.ne11 * c.ne10) / c.ne10;
    const long long i10 = row - i12 * c.ne11 * c.ne10 - i11 * c.ne10;
    const int row_index = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(idx) + i10 * c.idx_nb0 + i11 * c.idx_nb1 + i12 * c.idx_nb2);
    if (row_index < 0) {
        return;
    }

    const char * src_row = reinterpret_cast<const char *>(src0) +
        static_cast<long long>(row_index) * c.src0_nb1 + i11 * c.src0_nb2 + i12 * c.src0_nb3;
    char * dst_row = reinterpret_cast<char *>(dst) + i10 * c.dst_nb1 + i11 * c.dst_nb2 + i12 * c.dst_nb3;
    *reinterpret_cast<float *>(dst_row + col * sizeof(float)) =
        *reinterpret_cast<const float *>(src_row + col * sizeof(float));
}

extern "C" __global__ void hrx_get_rows_f32_nr1(
        const float * src0, const int * idx, float * dst,
        hrx_get_rows_f32_constants c) {
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    if (col >= c.nc) {
        return;
    }
    (void) c.nr;

    const int row_index = *reinterpret_cast<const int *>(reinterpret_cast<const char *>(idx));
    if (row_index < 0) {
        return;
    }

    const char * src_row = reinterpret_cast<const char *>(src0) + static_cast<long long>(row_index) * c.src0_nb1;
    dst[col] = *reinterpret_cast<const float *>(src_row + col * sizeof(float));
}
