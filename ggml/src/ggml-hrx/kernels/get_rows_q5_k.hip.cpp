#include <hip/hip_fp16.h>
#include <stdint.h>

struct hrx_block_q5_K {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct hrx_get_rows_q5_k_constants {
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

static __device__ __forceinline__ void hrx_get_scale_min_k4(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

extern "C" __global__ void hrx_get_rows_q5_k_f32(
        const hrx_block_q5_K * src0, const int * idx, float * dst,
        hrx_get_rows_q5_k_constants c) {
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
    const long long block_idx = col / 256;
    const int in_block = static_cast<int>(col - block_idx * 256);
    const int group = in_block / 32;
    const int lane = in_block & 31;
    const hrx_block_q5_K * block = reinterpret_cast<const hrx_block_q5_K *>(src_row) + block_idx;

    uint8_t sc = 0;
    uint8_t m = 0;
    hrx_get_scale_min_k4(group, block->scales, &sc, &m);

    const uint8_t low = block->qs[(group / 2) * 32 + lane];
    const float q = static_cast<float>(
        ((group & 1) ? (low >> 4) : (low & 0x0F)) +
        ((block->qh[lane] & (1u << group)) ? 16 : 0));
    const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
    const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);

    char * dst_row = reinterpret_cast<char *>(dst) + i10 * c.dst_nb1 + i11 * c.dst_nb2 + i12 * c.dst_nb3;
    *reinterpret_cast<float *>(dst_row + col * sizeof(float)) = d * q - min;
}
