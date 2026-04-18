#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_mul_add_add_f32_broadcast_constants {
    long long ne0;
    long long nrows;
    long long ne1;
    long long ne2;
    long long src1_ne0;
    long long src2_ne0;
    long long src3_ne0;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long src1_nb1;
    long long src1_nb2;
    long long src1_nb3;
    long long src2_nb1;
    long long src2_nb2;
    long long src2_nb3;
    long long src3_nb1;
    long long src3_nb2;
    long long src3_nb3;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
};

extern "C" __global__ void hrx_mul_add_add_f32_broadcast(
        const float * src0,
        const float * src1,
        const float * src2,
        const float * src3,
        float * dst,
        hrx_mul_add_add_f32_broadcast_constants c) {
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    if (col >= c.ne0 || row >= c.nrows) {
        return;
    }

    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;

    const char * src0_row = reinterpret_cast<const char *>(src0) +
        i1 * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3;
    const char * src1_row = reinterpret_cast<const char *>(src1) +
        (c.src1_nb1 == 0 ? 0 : i1 * c.src1_nb1) +
        (c.src1_nb2 == 0 ? 0 : i2 * c.src1_nb2) +
        (c.src1_nb3 == 0 ? 0 : i3 * c.src1_nb3);
    const char * src2_row = reinterpret_cast<const char *>(src2) +
        (c.src2_nb1 == 0 ? 0 : i1 * c.src2_nb1) +
        (c.src2_nb2 == 0 ? 0 : i2 * c.src2_nb2) +
        (c.src2_nb3 == 0 ? 0 : i3 * c.src2_nb3);
    const char * src3_row = reinterpret_cast<const char *>(src3) +
        (c.src3_nb1 == 0 ? 0 : i1 * c.src3_nb1) +
        (c.src3_nb2 == 0 ? 0 : i2 * c.src3_nb2) +
        (c.src3_nb3 == 0 ? 0 : i3 * c.src3_nb3);
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;

    const long long src1_col = c.src1_ne0 == 1 ? 0 : col;
    const long long src2_col = c.src2_ne0 == 1 ? 0 : col;
    const long long src3_col = c.src3_ne0 == 1 ? 0 : col;

    const float x = *reinterpret_cast<const float *>(src0_row + col * sizeof(float));
    const float y = *reinterpret_cast<const float *>(src1_row + src1_col * sizeof(float));
    const float z = *reinterpret_cast<const float *>(src2_row + src2_col * sizeof(float));
    const float w = *reinterpret_cast<const float *>(src3_row + src3_col * sizeof(float));
    *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = x * y + z + w;
}

static __device__ __forceinline__ float hrx_sigmoid_value_f32(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

extern "C" __global__ void hrx_sigmoid_mul_add_add_f32_broadcast(
        const float * src0,
        const float * sigmoid_src,
        const float * src2,
        const float * src3,
        float * dst,
        hrx_mul_add_add_f32_broadcast_constants c) {
    const long long col = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    if (col >= c.ne0 || row >= c.nrows) {
        return;
    }

    const long long i3 = row / (c.ne1 * c.ne2);
    const long long i2 = (row - i3 * c.ne1 * c.ne2) / c.ne1;
    const long long i1 = row - i3 * c.ne1 * c.ne2 - i2 * c.ne1;

    const char * src0_row = reinterpret_cast<const char *>(src0) +
        i1 * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3;
    const char * sigmoid_row = reinterpret_cast<const char *>(sigmoid_src) +
        (c.src1_nb1 == 0 ? 0 : i1 * c.src1_nb1) +
        (c.src1_nb2 == 0 ? 0 : i2 * c.src1_nb2) +
        (c.src1_nb3 == 0 ? 0 : i3 * c.src1_nb3);
    const char * src2_row = reinterpret_cast<const char *>(src2) +
        (c.src2_nb1 == 0 ? 0 : i1 * c.src2_nb1) +
        (c.src2_nb2 == 0 ? 0 : i2 * c.src2_nb2) +
        (c.src2_nb3 == 0 ? 0 : i3 * c.src2_nb3);
    const char * src3_row = reinterpret_cast<const char *>(src3) +
        (c.src3_nb1 == 0 ? 0 : i1 * c.src3_nb1) +
        (c.src3_nb2 == 0 ? 0 : i2 * c.src3_nb2) +
        (c.src3_nb3 == 0 ? 0 : i3 * c.src3_nb3);
    char * dst_row = reinterpret_cast<char *>(dst) +
        i1 * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3;

    const long long sigmoid_col = c.src1_ne0 == 1 ? 0 : col;
    const long long src2_col = c.src2_ne0 == 1 ? 0 : col;
    const long long src3_col = c.src3_ne0 == 1 ? 0 : col;

    const float x = *reinterpret_cast<const float *>(src0_row + col * sizeof(float));
    const float y = *reinterpret_cast<const float *>(sigmoid_row + sigmoid_col * sizeof(float));
    const float z = *reinterpret_cast<const float *>(src2_row + src2_col * sizeof(float));
    const float w = *reinterpret_cast<const float *>(src3_row + src3_col * sizeof(float));
    *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = x * hrx_sigmoid_value_f32(y) + z + w;
}
