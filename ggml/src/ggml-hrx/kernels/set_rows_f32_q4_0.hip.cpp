#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

struct hrx_set_rows_constants {
    long long nc;
    long long nr;
    long long ne02;
    long long ne03;
    long long ne1;
    long long ne11;
    long long ne12;
    long long src0_nb1;
    long long src0_nb2;
    long long src0_nb3;
    long long idx_nb0;
    long long idx_nb1;
    long long idx_nb2;
    long long dst_nb1;
    long long dst_nb2;
    long long dst_nb3;
};

struct hrx_block_q4_0 {
    unsigned short d;
    uint8_t qs[16];
};

static __device__ __forceinline__ uint8_t hrx_quantize_q4_0(float v) {
    const int q = static_cast<int>(v + 8.5f);
    return static_cast<uint8_t>(q < 0 ? 0 : (q > 15 ? 15 : q));
}

extern "C" __global__ void hrx_set_rows_f32_q4_0(
        const float * src0, const long long * idxs, hrx_block_q4_0 * dst,
        hrx_set_rows_constants c) {
    const long long linear = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 256 +
        __builtin_amdgcn_workitem_id_x();
    const long long nc_blocks = c.nc / 32;
    const long long total = nc_blocks * c.nr * c.ne02 * c.ne03;
    if (linear >= total) {
        return;
    }

    const long long ib0 = linear % nc_blocks;
    const long long i0 = ib0 * 32;
    const long long t1 = linear / nc_blocks;
    const long long i = t1 % c.nr;
    const long long t2 = t1 / c.nr;
    const long long i2 = t2 % c.ne02;
    const long long i3 = t2 / c.ne02;

    const long long i12 = i3 % c.ne12;
    const long long i11 = i2 % c.ne11;
    const long long row = *reinterpret_cast<const long long *>(
        reinterpret_cast<const char *>(idxs) + i * c.idx_nb0 + i11 * c.idx_nb1 + i12 * c.idx_nb2);
    if (row < 0 || row >= c.ne1) {
        return;
    }

    const char * src_base = reinterpret_cast<const char *>(src0) +
        i0 * static_cast<long long>(sizeof(float)) + i * c.src0_nb1 + i2 * c.src0_nb2 + i3 * c.src0_nb3;
    float amax = 0.0f;
    float max = 0.0f;
    for (int j = 0; j < 32; ++j) {
        const float v = *reinterpret_cast<const float *>(src_base + j * static_cast<long long>(sizeof(float)));
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            max = v;
        }
    }

    const float d = max / -8.0f;
    const float id = d ? 1.0f / d : 0.0f;
    hrx_block_q4_0 * block = reinterpret_cast<hrx_block_q4_0 *>(
        reinterpret_cast<char *>(dst) + ib0 * static_cast<long long>(sizeof(hrx_block_q4_0)) +
        row * c.dst_nb1 + i2 * c.dst_nb2 + i3 * c.dst_nb3);
    block->d = __half_as_ushort(__float2half(d));
    for (int j = 0; j < 16; ++j) {
        const float v0 = *reinterpret_cast<const float *>(src_base + j * static_cast<long long>(sizeof(float)));
        const float v1 = *reinterpret_cast<const float *>(src_base + (j + 16) * static_cast<long long>(sizeof(float)));
        const uint8_t q0 = hrx_quantize_q4_0(v0 * id);
        const uint8_t q1 = hrx_quantize_q4_0(v1 * id);
        block->qs[j] = q0 | (q1 << 4);
    }
}
