#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q6_K {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_wg(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (WG_SIZE <= warpSize) {
        return sum;
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < ((WG_SIZE + warpSize - 1) / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_reduce_wg16(float (&sum)[16], float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            sum[col] += __shfl_down(sum[col], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int col = 0; col < 16; ++col) {
            shared[wave + col * waves] = sum[col];
        }
    }
    __syncthreads();

    #pragma unroll
    for (int col = 0; col < 16; ++col) {
        sum[col] = lane < waves ? shared[lane + col * waves] : 0.0f;
    }
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            #pragma unroll
            for (int col = 0; col < 16; ++col) {
                sum[col] += __shfl_down(sum[col], offset);
            }
        }
    }
}

static __device__ __forceinline__ int hrx_q6_k_scale(
        const hrx_block_q6_K * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ float hrx_q6_k_dot4(
        const hrx_block_q6_K * block,
        const float * src,
        float d,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);
    float sum = 0.0f;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const int q = (ql | (qh << 4)) - 32;
        sum += d * static_cast<float>(q) * src[j];
    }

    return sum;
}

static __device__ __forceinline__ void hrx_q6_k_dot4_cols8_acc(
        const hrx_block_q6_K * block,
        const float * src0,
        long long k,
        float d,
        int group,
        int lane,
        float (&sum)[16],
        int sum_offset) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int ql_base = half * 64 + ((group_in_half & 1) ? 32 : 0) + lane;
    const int qh_base = half * 32 + lane;
    const int qh_shift = (group_in_half & 3) * 2;
    const bool high_nibble = group_in_half >= 2;

    const uint32_t ql_word =
        static_cast<uint32_t>(block->ql[ql_base]) |
        (static_cast<uint32_t>(block->ql[ql_base + 1]) << 8) |
        (static_cast<uint32_t>(block->ql[ql_base + 2]) << 16) |
        (static_cast<uint32_t>(block->ql[ql_base + 3]) << 24);
    const uint32_t qh_word =
        static_cast<uint32_t>(block->qh[qh_base]) |
        (static_cast<uint32_t>(block->qh[qh_base + 1]) << 8) |
        (static_cast<uint32_t>(block->qh[qh_base + 2]) << 16) |
        (static_cast<uint32_t>(block->qh[qh_base + 3]) << 24);

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int ql_shift = 8 * j + (high_nibble ? 4 : 0);
        const int ql = (ql_word >> ql_shift) & 0x0F;
        const int qh = (qh_word >> (8 * j + qh_shift)) & 0x03;
        const float q = d * static_cast<float>((ql | (qh << 4)) - 32);
        #pragma unroll
        for (int col = 0; col < 8; ++col) {
            sum[sum_offset + col] += q * src0[col * k + j];
        }
    }
}

static __device__ __forceinline__ uint32_t hrx_load_u32(const uint8_t * base) {
    return static_cast<uint32_t>(base[0]) |
        (static_cast<uint32_t>(base[1]) << 8) |
        (static_cast<uint32_t>(base[2]) << 16) |
        (static_cast<uint32_t>(base[3]) << 24);
}

static __device__ __forceinline__ float hrx_q6_k_dot16(
        const hrx_block_q6_K * block,
        const float * src,
        int itid) {
    const int v_im = itid >> 3;
    const int v_in = itid & 7;
    const int l0 = 4 * v_in;
    const int is = v_in >> 2;

    const int ql_offset = 64 * v_im + l0;
    const int qh_offset = 32 * v_im + l0;
    const int s_offset = 8 * v_im + is;
    const int y_offset = 128 * v_im + l0;

    const uint32_t ql0 = hrx_load_u32(block->ql + ql_offset);
    const uint32_t ql32 = hrx_load_u32(block->ql + ql_offset + 32);
    const uint32_t qh = hrx_load_u32(block->qh + qh_offset);

    const uint32_t q0 = (ql0 & 0x0F0F0F0Fu) | ((qh & 0x03030303u) << 4);
    const uint32_t q1 = (ql32 & 0x0F0F0F0Fu) | ((qh & 0x0C0C0C0Cu) << 2);
    const uint32_t q2 = ((ql0 >> 4) & 0x0F0F0F0Fu) | (qh & 0x30303030u);
    const uint32_t q3 = ((ql32 >> 4) & 0x0F0F0F0Fu) | ((qh & 0xC0C0C0C0u) >> 2);

    const float4 by0 = *reinterpret_cast<const float4 *>(src + y_offset);
    const float4 by32 = *reinterpret_cast<const float4 *>(src + y_offset + 32);
    const float4 by64 = *reinterpret_cast<const float4 *>(src + y_offset + 64);
    const float4 by96 = *reinterpret_cast<const float4 *>(src + y_offset + 96);

    const float sx =
        (static_cast<float>((q0 >> 0) & 0xFFu) - 32.0f) * by0.x +
        (static_cast<float>((q0 >> 8) & 0xFFu) - 32.0f) * by0.y +
        (static_cast<float>((q0 >> 16) & 0xFFu) - 32.0f) * by0.z +
        (static_cast<float>((q0 >> 24) & 0xFFu) - 32.0f) * by0.w;
    const float sy =
        (static_cast<float>((q1 >> 0) & 0xFFu) - 32.0f) * by32.x +
        (static_cast<float>((q1 >> 8) & 0xFFu) - 32.0f) * by32.y +
        (static_cast<float>((q1 >> 16) & 0xFFu) - 32.0f) * by32.z +
        (static_cast<float>((q1 >> 24) & 0xFFu) - 32.0f) * by32.w;
    const float sz =
        (static_cast<float>((q2 >> 0) & 0xFFu) - 32.0f) * by64.x +
        (static_cast<float>((q2 >> 8) & 0xFFu) - 32.0f) * by64.y +
        (static_cast<float>((q2 >> 16) & 0xFFu) - 32.0f) * by64.z +
        (static_cast<float>((q2 >> 24) & 0xFFu) - 32.0f) * by64.w;
    const float sw =
        (static_cast<float>((q3 >> 0) & 0xFFu) - 32.0f) * by96.x +
        (static_cast<float>((q3 >> 8) & 0xFFu) - 32.0f) * by96.y +
        (static_cast<float>((q3 >> 16) & 0xFFu) - 32.0f) * by96.z +
        (static_cast<float>((q3 >> 24) & 0xFFu) - 32.0f) * by96.w;

    const float d = __half2float(__ushort_as_half(block->d));
    return d * (
        sx * static_cast<float>(block->scales[s_offset]) +
        sy * static_cast<float>(block->scales[s_offset + 2]) +
        sz * static_cast<float>(block->scales[s_offset + 4]) +
        sw * static_cast<float>(block->scales[s_offset + 6]));
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_q6_k_f32_impl(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row_blocks = src0 + row * blocks_per_row;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q6_K * block = row_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, lane));

        sum += hrx_q6_k_dot4(block, src1_col + src_base, d, group, lane);
    }

    sum = hrx_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_f32_impl<256>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_wg128_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_f32_impl<128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_wg64_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_q6_k_f32_impl<64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_rows2_cols1_wg32_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows || col >= cols) {
        return;
    }

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row0_blocks = src0 + row0 * blocks_per_row;
    const hrx_block_q6_K * row1_blocks = row0_blocks + blocks_per_row;
    const float * src1_col = src1 + col * k;

    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const float * src_block = src1_col + block_idx * 256;
        sum0 += hrx_q6_k_dot16(row0_blocks + block_idx, src_block, itid);
        if (row0 + 1 < rows) {
            sum1 += hrx_q6_k_dot16(row1_blocks + block_idx, src_block, itid);
        }
    }

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }

    if (tid == 0) {
        dst[col * rows + row0] = sum0;
        if (row0 + 1 < rows) {
            dst[col * rows + row0 + 1] = sum1;
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_rows2_cols8_wg128_f32(
        const hrx_block_q6_K * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[16 * (128 / 32)];

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K * row0_blocks = src0 + row0 * blocks_per_row;
    const hrx_block_q6_K * row1_blocks = row0_blocks + blocks_per_row;
    const float * src1_col0 = src1 + col0 * k;
    float sum[16] = {};

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 2) {
        const hrx_block_q6_K * block0 = row0_blocks + block_idx;
        const hrx_block_q6_K * block1 = row1_blocks + block_idx;
        const long long src_base = block_idx * 256 + in_block_base;
        const float d0 = __half2float(__ushort_as_half(block0->d)) *
            static_cast<float>(hrx_q6_k_scale(block0, group, lane));
        const float d1 = __half2float(__ushort_as_half(block1->d)) *
            static_cast<float>(hrx_q6_k_scale(block1, group, lane));

        hrx_q6_k_dot4_cols8_acc(block0, src1_col0 + src_base, k, d0, group, lane, sum, 0);
        hrx_q6_k_dot4_cols8_acc(block1, src1_col0 + src_base, k, d1, group, lane, sum, 8);
    }

    hrx_reduce_wg16<128>(sum, sumsh);

    if (tid == 0) {
        #pragma unroll
        for (int col = 0; col < 8; ++col) {
            dst[(col0 + col) * rows + row0] = sum[col];
            dst[(col0 + col) * rows + row0 + 1] = sum[8 + col];
        }
    }
}
