#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q6_K_q8_1_lhs {
    uint8_t ql[128];
    uint8_t qh[64];
    int8_t scales[16];
    unsigned short d;
};

struct hrx_block_q8_1_rhs_q6 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_rhs_q6 {
    unsigned short ds[8];
    int qs[32];
};

static __device__ __forceinline__ float hrx_reduce_256_q6_q8_1(float sum, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset);
    }
    if (lane == 0) {
        shared[wave] = sum;
    }
    __syncthreads();

    sum = lane < (256 / warpSize) ? shared[lane] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum += __shfl_down(sum, offset);
        }
    }
    return sum;
}

static __device__ __forceinline__ int hrx_q6_k_value(
        const hrx_block_q6_K_q8_1_lhs * block,
        int in_block) {
    const int half = in_block / 128;
    const int idx = in_block - half * 128;
    const int lane = idx & 31;
    const int ql_base = half * 64;
    const int qh_base = half * 32;

    int q = 0;
    if (idx < 32) {
        q = (block->ql[ql_base + lane] & 0x0F) | (((block->qh[qh_base + lane] >> 0) & 3) << 4);
    } else if (idx < 64) {
        q = (block->ql[ql_base + lane + 32] & 0x0F) | (((block->qh[qh_base + lane] >> 2) & 3) << 4);
    } else if (idx < 96) {
        q = (block->ql[ql_base + lane] >> 4) | (((block->qh[qh_base + lane] >> 4) & 3) << 4);
    } else {
        q = (block->ql[ql_base + lane + 32] >> 4) | (((block->qh[qh_base + lane] >> 6) & 3) << 4);
    }
    return q - 32;
}

static __device__ __forceinline__ int hrx_q6_k_scale(
        const hrx_block_q6_K_q8_1_lhs * block,
        int group,
        int lane) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    return static_cast<int>(block->scales[half * 8 + group_in_half * 2 + lane / 16]);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1(int q0, int q1, int q2, int q3, const int8_t * rhs) {
    const unsigned int qpack =
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q0))) << 0) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q1))) << 8) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q2))) << 16) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q3))) << 24);
    const int rpack = *reinterpret_cast<const int *>(rhs);
    return __builtin_amdgcn_sudot4(true, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1_packed(int q0, int q1, int q2, int q3, int rpack) {
    const unsigned int qpack =
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q0))) << 0) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q1))) << 8) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q2))) << 16) |
        (static_cast<unsigned int>(static_cast<unsigned char>(static_cast<int8_t>(q3))) << 24);
    return __builtin_amdgcn_sudot4(true, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_sdot4_q6_q8_1_qpack(int qpack, int rpack) {
    return __builtin_amdgcn_sudot4(true, qpack, true, rpack, 0, false);
}

static __device__ __forceinline__ int hrx_q6_k_pack4(
        const hrx_block_q6_K_q8_1_lhs * block,
        int group,
        int iqs) {
    const int half = group >> 2;
    const int group_in_half = group & 3;
    const int lane = iqs * 4;
    const int ql_base = half * 64 + lane + ((group_in_half & 1) ? 32 : 0);
    const int qh_base = half * 32 + lane;
    const int ql_shift = (group_in_half >> 1) * 4;
    const int qh_shift = group_in_half * 2;

    const uint32_t ql = *reinterpret_cast<const uint32_t *>(block->ql + ql_base);
    const uint32_t qh = *reinterpret_cast<const uint32_t *>(block->qh + qh_base);
    const uint32_t qu =
        ((ql >> ql_shift) & 0x0F0F0F0Fu) |
        (((qh >> qh_shift) & 0x03030303u) << 4);
    const uint32_t sign_extend = ((~qu) & 0x20202020u) * 7u;
    return static_cast<int>((qu & 0x1F1F1F1Fu) | sign_extend);
}

struct hrx_q6_k_mmqv_a_cache {
    int qs[8];
    float d[2];
};

struct hrx_q8_1_mmqv_b_cache_q6 {
    int qs[8];
    float d;
};

static __device__ __forceinline__ void hrx_q6_k_mmqv_load_a(
        hrx_q6_k_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q6_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q6_K_q8_1_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = hrx_q6_k_pack4(block, group, iqs);
    if (iqs == 0 || iqs == 4) {
        buf_a[buf_idx].d[iqs >> 2] =
            __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, iqs * 4));
    }
}

static __device__ __forceinline__ void hrx_q6_k_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache_q6 * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q6 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK_STEP = 4;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = 32;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;

    static_assert(WNITER == 4, "unexpected Q6 MMQ 128x64 tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Vulkan large Q6 MMQ subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q6_k_mmqv_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmqv_b_cache_q6 buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;

    float sum[WNITER * TM * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadstride_a = BLOCK_SIZE * LOAD_VEC_A / 32;
        for (int r = loadc_a; r < BM; r += loadstride_a) {
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q6_k_mmqv_load_a(
                    buf_a,
                    k_step * BM + r,
                    src0,
                    row_base + r,
                    kb_base + k_step,
                    loadr_a,
                    blocks_per_row);
            }
        }

        const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
        const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
        const int loadstride_b = BLOCK_SIZE * LOAD_VEC_B / 32;
        for (int c = loadc_b; c < BN; c += loadstride_b) {
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q6_k_mmqv_load_b(
                    buf_b,
                    k_step * BN + c,
                    src1,
                    col_base + c,
                    kb_base + k_step,
                    loadr_b,
                    q8_blocks_per_col);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            hrx_q6_k_mmqv_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    hrx_q8_1_mmqv_b_cache_q6 cache_b =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum0 = 0;
                        int qsum1 = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 4; ++iqs) {
                            qsum0 += hrx_sdot4_q6_q8_1_qpack(
                                cache_a[cr].qs[iqs], cache_b.qs[iqs]);
                        }
                        #pragma unroll
                        for (int iqs = 4; iqs < 8; ++iqs) {
                            qsum1 += hrx_sdot4_q6_q8_1_qpack(
                                cache_a[cr].qs[iqs], cache_b.qs[iqs]);
                        }
                        sum[(wsic * TM + cr) * TN + cc] += cache_b.d *
                            (cache_a[cr].d[0] * static_cast<float>(qsum0) +
                             cache_a[cr].d[1] * static_cast<float>(qsum1));
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int wsic = 0; wsic < WNITER; ++wsic) {
        #pragma unroll
        for (int cr = 0; cr < TM; ++cr) {
            const long long row = row_base + warp_r * WM + tiwr * TM + cr;
            #pragma unroll
            for (int cc = 0; cc < TN; ++cc) {
                const long long col = col_base + warp_c * WN + wsic * WSUBN + tiwc * TN + cc;
                dst[col * rows + row] = sum[(wsic * TM + cr) * TN + cc];
            }
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int COLS_PER_THREAD = 8;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int col_lane = static_cast<int>(tid >> 5);
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM +
        static_cast<long long>(tid & 31u);
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN +
        static_cast<long long>(col_lane * COLS_PER_THREAD);
    if (row >= rows || col_base + COLS_PER_THREAD - 1 >= cols) {
        return;
    }

    __shared__ int b_qs[BN][8];
    __shared__ unsigned short b_d[BN];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const hrx_block_q6_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 128) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
            const hrx_block_q8_1_x4_rhs_q6 * rhs = src1 + (linear_block >> 2);
            const int inner = static_cast<int>(linear_block & 3);
            b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
            if (iqs == 0) {
                b_d[c] = rhs->ds[inner * 2 + 0];
            }
        }
        __syncthreads();

        const hrx_block_q6_K_q8_1_lhs * block = row_blocks + (kb >> 3);
        const int group = static_cast<int>(kb & 7);
        const float d0 = __half2float(__ushort_as_half(block->d));

        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const int in_block_base = group * 32 + iqs * 4;
            const int q0 = hrx_q6_k_value(block, in_block_base + 0);
            const int q1 = hrx_q6_k_value(block, in_block_base + 1);
            const int q2 = hrx_q6_k_value(block, in_block_base + 2);
            const int q3 = hrx_q6_k_value(block, in_block_base + 3);
            const float d = d0 * static_cast<float>(hrx_q6_k_scale(block, group, iqs * 4));

            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                const int c = col_lane * COLS_PER_THREAD + col;
                const int qsum = hrx_sdot4_q6_q8_1_packed(q0, q1, q2, q3, b_qs[c][iqs]);
                sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum);
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        dst[(col_base + col) * rows + row] = sum[col];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q6_k_q8_1_f32(
        const hrx_block_q6_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q6 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[256];

    const long long blocks_per_row = k / 256;
    const hrx_block_q6_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const hrx_block_q8_1_rhs_q6 * src1_col = src1 + col * (k / 32);
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const int in_block_base = group * 32 + lane;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const hrx_block_q6_K_q8_1_lhs * block = row_blocks + block_idx;
        const hrx_block_q8_1_rhs_q6 * rhs = src1_col + block_idx * 8 + group;

        const int qsum = hrx_sdot4_q6_q8_1(
            hrx_q6_k_value(block, in_block_base + 0),
            hrx_q6_k_value(block, in_block_base + 1),
            hrx_q6_k_value(block, in_block_base + 2),
            hrx_q6_k_value(block, in_block_base + 3),
            rhs->qs + lane);

        const float d = __half2float(__ushort_as_half(block->d)) *
            static_cast<float>(hrx_q6_k_scale(block, group, lane));
        const float d8 = __half2float(__ushort_as_half(rhs->d));
        sum += d * d8 * static_cast<float>(qsum);
    }

    sum = hrx_reduce_256_q6_q8_1(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}
