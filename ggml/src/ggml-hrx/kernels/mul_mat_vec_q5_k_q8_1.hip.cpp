#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q5_K_q8_1_lhs {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qh[32];
    uint8_t qs[128];
};

struct hrx_block_q8_1_rhs_q5 {
    unsigned short d;
    unsigned short s;
    int8_t qs[32];
};

struct hrx_block_q8_1_x4_rhs_q5 {
    unsigned short ds[8];
    int qs[32];
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_q5_q8_1(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ float hrx_reduce_256_q5_q8_1(float sum, float * shared) {
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

static __device__ __forceinline__ int hrx_sudot4_q5_q8_1(uint32_t qpack, int rpack) {
    return __builtin_amdgcn_sudot4(false, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ uint32_t hrx_q5_k_pack4(
        const hrx_block_q5_K_q8_1_lhs * block,
        int group,
        int iqs) {
    const int qs_base = (group >> 1) * 32 + iqs * 4;
    uint32_t qs = *reinterpret_cast<const uint32_t *>(block->qs + qs_base);
    qs = (qs >> ((group & 1) * 4)) & 0x0F0F0F0Fu;

    const uint32_t qh = *reinterpret_cast<const uint32_t *>(block->qh + iqs * 4);
    return qs | (((qh >> group) & 0x01010101u) << 4);
}

struct hrx_q5_k_mmqv_a_cache {
    int qs[8];
    float d;
    float min;
};

struct hrx_q8_1_mmqv_b_cache {
    int qs[8];
    float d;
    float s;
};

static __device__ __forceinline__ void hrx_q5_k_mmqv_load_a(
        hrx_q5_k_mmqv_a_cache * buf_a,
        int buf_idx,
        const hrx_block_q5_K_q8_1_lhs * src0,
        long long row,
        long long kb,
        int iqs,
        long long blocks_per_row) {
    const hrx_block_q5_K_q8_1_lhs * block = src0 + row * blocks_per_row + (kb >> 3);
    const int group = static_cast<int>(kb & 7);
    buf_a[buf_idx].qs[iqs] = static_cast<int>(hrx_q5_k_pack4(block, group, iqs));
    if (iqs == 0) {
        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        buf_a[buf_idx].d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        buf_a[buf_idx].min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
    }
}

static __device__ __forceinline__ void hrx_q5_k_mmqv_load_b(
        hrx_q8_1_mmqv_b_cache * buf_b,
        int buf_idx,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
        long long col,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    const long long linear_block = col * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_rhs_q5 * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        buf_b[buf_idx].qs[iqs_vec4 * 4 + j] = rhs->qs[inner * 8 + iqs_vec4 * 4 + j];
    }
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 0]));
        buf_b[buf_idx].s = __half2float(__ushort_as_half(rhs->ds[inner * 2 + 1]));
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 256;
    constexpr int WARP = 64;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WMITER = 1;
    constexpr int TM = 4;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;

    static_assert(WNITER == 8, "unexpected Vulkan large Q5 MMQ tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Vulkan large Q5 MMQ subtile shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int warp_i = static_cast<int>(tid / WARP);
    const int tiw = static_cast<int>(tid % WARP);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const int warp_r = warp_i % (BM / WM);
    const int warp_c = warp_i / (BM / WM);

    __shared__ hrx_q5_k_mmqv_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_mmqv_b_cache buf_b[BN * BK_STEP];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;

    float sum[WNITER * TM * TN] = {};

    for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
        const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
        const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
        const int loadstride_a = BLOCK_SIZE * LOAD_VEC_A / 32;
        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            for (int r = loadc_a; r < BM; r += loadstride_a) {
                hrx_q5_k_mmqv_load_a(
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
        #pragma unroll
        for (int k_step = 0; k_step < BK_STEP; ++k_step) {
            for (int c = loadc_b; c < BN; c += loadstride_b) {
                hrx_q5_k_mmqv_load_b(
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
            hrx_q5_k_mmqv_a_cache cache_a[TM];
            #pragma unroll
            for (int cr = 0; cr < TM; ++cr) {
                cache_a[cr] = buf_a[k_step * BM + warp_r * WM + tiwr * TM + cr];
            }

            #pragma unroll
            for (int wsic = 0; wsic < WNITER; ++wsic) {
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    hrx_q8_1_mmqv_b_cache cache_b =
                        buf_b[k_step * BN + warp_c * WN + wsic * WSUBN + tiwc * TN + cc];
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        int qsum = 0;
                        #pragma unroll
                        for (int iqs = 0; iqs < 8; ++iqs) {
                            qsum += hrx_sudot4_q5_q8_1(
                                static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b.qs[iqs]);
                        }
                        sum[(wsic * TN + cc) * TM + cr] +=
                            cache_a[cr].d * cache_b.d * static_cast<float>(qsum) -
                            cache_a[cr].min * cache_b.s;
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
                dst[col * rows + row] = sum[(wsic * TN + cc) * TM + cr];
            }
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q5 * src1,
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
    __shared__ unsigned short b_s[BN];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const hrx_block_q5_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 128) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const hrx_block_q8_1_rhs_q5 * rhs =
                src1 + (static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN + c) * q8_blocks_per_col + kb;
            b_qs[c][iqs] = *reinterpret_cast<const int *>(rhs->qs + iqs * 4);
            if (iqs == 0) {
                b_d[c] = rhs->d;
                b_s[c] = rhs->s;
            }
        }
        __syncthreads();

        const hrx_block_q5_K_q8_1_lhs * block = row_blocks + (kb >> 3);
        const int group = static_cast<int>(kb & 7);

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);

        int qsum[COLS_PER_THREAD] = {};
        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const uint32_t qpack = hrx_q5_k_pack4(block, group, iqs);
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum[col] += hrx_sudot4_q5_q8_1(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum[col]) -
                min * __half2float(__ushort_as_half(b_s[c]));
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        dst[(col_base + col) * rows + row] = sum[col];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
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
    __shared__ unsigned short b_s[BN];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const hrx_block_q5_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 128) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
            const hrx_block_q8_1_x4_rhs_q5 * rhs = src1 + (linear_block >> 2);
            const int inner = static_cast<int>(linear_block & 3);
            b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
            if (iqs == 0) {
                b_d[c] = rhs->ds[inner * 2 + 0];
                b_s[c] = rhs->ds[inner * 2 + 1];
            }
        }
        __syncthreads();

        const hrx_block_q5_K_q8_1_lhs * block = row_blocks + (kb >> 3);
        const int group = static_cast<int>(kb & 7);

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);

        int qsum[COLS_PER_THREAD] = {};
        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const uint32_t qpack = hrx_q5_k_pack4(block, group, iqs);
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum[col] += hrx_sudot4_q5_q8_1(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum[col]) -
                min * __half2float(__ushort_as_half(b_s[c]));
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        dst[(col_base + col) * rows + row] = sum[col];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_x4_rhs_q5 * src1,
        float * dst,
        long long k, long long rows, long long cols) {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int COLS_PER_THREAD = 16;

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const int row_lane = static_cast<int>(tid & 63u);
    const int col_lane = static_cast<int>(tid >> 6);
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM + row_lane;
    const long long col_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN +
        static_cast<long long>(col_lane * COLS_PER_THREAD);
    if (row >= rows || col_base + COLS_PER_THREAD - 1 >= cols) {
        return;
    }

    __shared__ int b_qs[BN][8];
    __shared__ unsigned short b_d[BN];
    __shared__ unsigned short b_s[BN];

    const long long blocks_per_row = k / 256;
    const long long q8_blocks_per_col = k / 32;
    const long long col_block_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const hrx_block_q5_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    float sum[COLS_PER_THREAD] = {};

    for (long long kb = 0; kb < q8_blocks_per_col; ++kb) {
        #pragma unroll
        for (int load_idx = static_cast<int>(tid); load_idx < BN * 8; load_idx += 256) {
            const int c = load_idx >> 3;
            const int iqs = load_idx & 7;
            const long long linear_block = (col_block_base + c) * q8_blocks_per_col + kb;
            const hrx_block_q8_1_x4_rhs_q5 * rhs = src1 + (linear_block >> 2);
            const int inner = static_cast<int>(linear_block & 3);
            b_qs[c][iqs] = rhs->qs[inner * 8 + iqs];
            if (iqs == 0) {
                b_d[c] = rhs->ds[inner * 2 + 0];
                b_s[c] = rhs->ds[inner * 2 + 1];
            }
        }
        __syncthreads();

        const hrx_block_q5_K_q8_1_lhs * block = row_blocks + (kb >> 3);
        const int group = static_cast<int>(kb & 7);

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);

        int qsum[COLS_PER_THREAD] = {};
        #pragma unroll
        for (int iqs = 0; iqs < 8; ++iqs) {
            const uint32_t qpack = hrx_q5_k_pack4(block, group, iqs);
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                qsum[col] += hrx_sudot4_q5_q8_1(qpack, b_qs[col_lane * COLS_PER_THREAD + col][iqs]);
            }
        }

        #pragma unroll
        for (int col = 0; col < COLS_PER_THREAD; ++col) {
            const int c = col_lane * COLS_PER_THREAD + col;
            sum[col] += d * __half2float(__ushort_as_half(b_d[c])) * static_cast<float>(qsum[col]) -
                min * __half2float(__ushort_as_half(b_s[c]));
        }

        __syncthreads();
    }

    #pragma unroll
    for (int col = 0; col < COLS_PER_THREAD; ++col) {
        dst[(col_base + col) * rows + row] = sum[col];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_q5_k_q8_1_f32(
        const hrx_block_q5_K_q8_1_lhs * src0,
        const hrx_block_q8_1_rhs_q5 * src1,
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
    const hrx_block_q5_K_q8_1_lhs * row_blocks = src0 + row * blocks_per_row;
    const hrx_block_q8_1_rhs_q5 * src1_col = src1 + col * (k / 32);
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const hrx_block_q5_K_q8_1_lhs * block = row_blocks + block_idx;
        const hrx_block_q8_1_rhs_q5 * rhs = src1_col + block_idx * 8 + group;

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_q5_q8_1(group, block->scales, &sc, &m);

        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const float d8 = __half2float(__ushort_as_half(rhs->d));
        const float rhs_sum = __half2float(__ushort_as_half(rhs->s));
        const int qs_base = (group >> 1) * 32 + lane;

        int qsum = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t low = block->qs[qs_base + j];
            const int q =
                ((group & 1) ? (low >> 4) : (low & 0x0F)) +
                ((block->qh[lane + j] & (1u << group)) ? 16 : 0);
            qsum += q * static_cast<int>(rhs->qs[lane + j]);
        }
        sum += d * d8 * static_cast<float>(qsum);
        if (lane == 0) {
            sum -= min * rhs_sum;
        }
    }

    sum = hrx_reduce_256_q5_q8_1(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}
