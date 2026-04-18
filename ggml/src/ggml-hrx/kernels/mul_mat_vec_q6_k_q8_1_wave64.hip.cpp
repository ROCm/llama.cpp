#include "mul_mat_vec_q6_k_q8_1_common.hip.inc"

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
