#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_moe_mmq {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_block_q8_1_x4_moe_mmq {
    unsigned short ds[8];
    int qs[32];
};

struct hrx_mul_mat_id_q4_k_grouped_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long route_capacity;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long dst_nb1;
    long long dst_nb2;
};

struct hrx_mul_mat_id_q4_k_swiglu_grouped_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long route_capacity;
    long long gate_nb1;
    long long gate_nb2;
    long long up_nb1;
    long long up_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long dst_nb1;
    long long dst_nb2;
};

struct hrx_q4_k_moe_a_cache {
    int qs[8];
    float d;
    float min;
};

struct hrx_q8_1_moe_b_cache {
    int qs[8];
    float d;
    float s;
    uint32_t route;
};

struct hrx_q4_k_moe_a_pending {
    uint32_t qs_raw;
    unsigned short d;
    unsigned short dmin;
    uint8_t scale;
    uint8_t min;
    int shift;
    bool valid;
};

struct hrx_q8_1_moe_b_pending {
    int4 qs;
    unsigned short d;
    unsigned short s;
    uint32_t route;
    bool valid;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_moe_mmq(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static __device__ __forceinline__ int hrx_udot4_q4_q8_1_moe_mmq(uint32_t qpack, int rpack) {
    return __builtin_amdgcn_sudot4(false, static_cast<int>(qpack), true, rpack, 0, false);
}

static __device__ __forceinline__ hrx_q4_k_moe_a_pending hrx_q4_k_moe_mmq_fetch_a(
        const hrx_block_q4_K_moe_mmq * src0,
        long long row,
        long long kb,
        int iqs,
        long long rows,
        long long src0_nb1,
        long long src0_nb2,
        long long expert) {
    hrx_q4_k_moe_a_pending pending = {};
    pending.valid = row < rows;
    if (!pending.valid) {
        return pending;
    }

    const long long block_idx = kb >> 3;
    const int group = static_cast<int>(kb & 7);
    const char * expert_base = reinterpret_cast<const char *>(src0) + expert * src0_nb2;
    const hrx_block_q4_K_moe_mmq * block = reinterpret_cast<const hrx_block_q4_K_moe_mmq *>(
        expert_base + row * src0_nb1 + block_idx * sizeof(hrx_block_q4_K_moe_mmq));
    const int qs_base = (group >> 1) * 32 + iqs * 4;
    pending.qs_raw = *reinterpret_cast<const uint32_t *>(block->qs + qs_base);
    pending.shift = (group & 1) * 4;

    if (iqs == 0) {
        pending.d = block->d;
        pending.dmin = block->dmin;
        hrx_get_scale_min_k4_moe_mmq(group, block->scales, &pending.scale, &pending.min);
    }
    return pending;
}

static __device__ __forceinline__ void hrx_q4_k_moe_mmq_commit_a(
        hrx_q4_k_moe_a_cache * buf_a,
        int buf_idx,
        const hrx_q4_k_moe_a_pending & pending,
        int iqs) {
    if (!pending.valid) {
        buf_a[buf_idx].qs[iqs] = 0;
        if (iqs == 0) {
            buf_a[buf_idx].d = 0.0f;
            buf_a[buf_idx].min = 0.0f;
        }
        return;
    }

    buf_a[buf_idx].qs[iqs] = static_cast<int>((pending.qs_raw >> pending.shift) & 0x0F0F0F0Fu);
    if (iqs == 0) {
        buf_a[buf_idx].d = __half2float(__ushort_as_half(pending.d)) * static_cast<float>(pending.scale);
        buf_a[buf_idx].min = __half2float(__ushort_as_half(pending.dmin)) * static_cast<float>(pending.min);
    }
}

static __device__ __forceinline__ hrx_q8_1_moe_b_pending hrx_q8_1_moe_mmq_fetch_b(
        const hrx_block_q8_1_x4_moe_mmq * src1,
        const uint32_t * expert_routes,
        uint32_t count,
        uint32_t route_index,
        long long kb,
        int iqs_vec4,
        long long q8_blocks_per_col) {
    hrx_q8_1_moe_b_pending pending = {};
    pending.valid = route_index < count;
    pending.route = pending.valid ? expert_routes[route_index] : 0u;
    if (!pending.valid) {
        return pending;
    }

    const long long linear_block = static_cast<long long>(pending.route) * q8_blocks_per_col + kb;
    const hrx_block_q8_1_x4_moe_mmq * rhs = src1 + (linear_block >> 2);
    const int inner = static_cast<int>(linear_block & 3);
    pending.qs = *reinterpret_cast<const int4 *>(&rhs->qs[inner * 8 + iqs_vec4 * 4]);
    if (iqs_vec4 == 0) {
        pending.d = rhs->ds[inner * 2 + 0];
        pending.s = rhs->ds[inner * 2 + 1];
    }
    return pending;
}

static __device__ __forceinline__ void hrx_q8_1_moe_mmq_commit_b(
        hrx_q8_1_moe_b_cache * buf_b,
        int buf_idx,
        const hrx_q8_1_moe_b_pending & pending,
        int iqs_vec4) {
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 0] = pending.valid ? pending.qs.x : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 1] = pending.valid ? pending.qs.y : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 2] = pending.valid ? pending.qs.z : 0;
    buf_b[buf_idx].qs[iqs_vec4 * 4 + 3] = pending.valid ? pending.qs.w : 0;
    if (iqs_vec4 == 0) {
        buf_b[buf_idx].d = pending.valid ? __half2float(__ushort_as_half(pending.d)) : 0.0f;
        buf_b[buf_idx].s = pending.valid ? __half2float(__ushort_as_half(pending.s)) : 0.0f;
        buf_b[buf_idx].route = pending.route;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32(
        const hrx_block_q4_K_moe_mmq * src0,
        const hrx_block_q8_1_x4_moe_mmq * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_grouped_constants c) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 64;
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
    constexpr int LOAD_STRIDE_A = BLOCK_SIZE * LOAD_VEC_A / 32;
    constexpr int LOAD_STRIDE_B = BLOCK_SIZE * LOAD_VEC_B / 32;
    constexpr int LOADS_A = BM / LOAD_STRIDE_A;
    constexpr int LOADS_B = BN / LOAD_STRIDE_B;

    static_assert(WNITER == 4, "unexpected Q4 MoE MMQ tile shape");
    static_assert(WSUBM == 64 && WSUBN == 8, "unexpected Q4 MoE MMQ subtile shape");
    static_assert(LOADS_A == 8 && LOADS_B == 1, "unexpected Q4 MoE MMQ load shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const uint32_t route_base0 = static_cast<uint32_t>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    if (expert >= c.n_experts) {
        return;
    }
    const uint32_t count = counts[expert];
    if (route_base0 >= count || row_base >= c.rows) {
        return;
    }

    const int tiw = static_cast<int>(tid);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const uint32_t * expert_routes = routes + expert * c.route_capacity;
    const long long q8_blocks_per_col = c.k / 32;
    const uint32_t route_tile_span = static_cast<uint32_t>(((c.n_tokens + BN - 1) / BN) * BN);

    __shared__ hrx_q4_k_moe_a_cache buf_a[BM * BK_STEP];
    __shared__ hrx_q8_1_moe_b_cache buf_b[BN * BK_STEP];

    for (uint32_t route_base = route_base0; route_base < count; route_base += route_tile_span) {
        float sum[WNITER * TM * TN] = {};

        for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
            const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
            const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
            const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
            const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q4_k_moe_a_pending pending_a[LOADS_A];
                hrx_q8_1_moe_b_pending pending_b[LOADS_B];
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_A; ++load_i) {
                    const int r = loadc_a + load_i * LOAD_STRIDE_A;
                    pending_a[load_i] = hrx_q4_k_moe_mmq_fetch_a(
                        src0,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        c.rows,
                        c.src0_nb1,
                        c.src0_nb2,
                        expert);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    pending_b[load_i] = hrx_q8_1_moe_mmq_fetch_b(
                        src1,
                        expert_routes,
                        count,
                        route_base + col,
                        kb_base + k_step,
                        loadr_b,
                        q8_blocks_per_col);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_A; ++load_i) {
                    const int r = loadc_a + load_i * LOAD_STRIDE_A;
                    hrx_q4_k_moe_mmq_commit_a(buf_a, k_step * BM + r, pending_a[load_i], loadr_a);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    hrx_q8_1_moe_mmq_commit_b(buf_b, k_step * BN + col, pending_b[load_i], loadr_b);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q4_k_moe_a_cache cache_a[TM];
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    cache_a[cr] = buf_a[k_step * BM + tiwr * TM + cr];
                }
                #pragma unroll
                for (int wsic = 0; wsic < WNITER; ++wsic) {
                    hrx_q8_1_moe_b_cache cache_b[TN];
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        cache_b[cc] = buf_b[k_step * BN + wsic * WSUBN + tiwc * TN + cc];
                    }
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        #pragma unroll
                        for (int cc = 0; cc < TN; ++cc) {
                            int qsum = 0;
                            #pragma unroll
                            for (int iqs = 0; iqs < 8; ++iqs) {
                                qsum += hrx_udot4_q4_q8_1_moe_mmq(
                                    static_cast<uint32_t>(cache_a[cr].qs[iqs]), cache_b[cc].qs[iqs]);
                            }
                            sum[(wsic * TM + cr) * TN + cc] +=
                                cache_a[cr].d * cache_b[cc].d * static_cast<float>(qsum) -
                                cache_a[cr].min * cache_b[cc].s;
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
                const long long row = row_base + tiwr * TM + cr;
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const int col = wsic * WSUBN + tiwc * TN + cc;
                    const uint32_t route_index = route_base + static_cast<uint32_t>(col);
                    if (row < c.rows && route_index < count) {
                        const uint32_t route = buf_b[col].route;
                        const long long id = static_cast<long long>(route % static_cast<uint32_t>(c.n_ids));
                        const long long token = static_cast<long long>(route / static_cast<uint32_t>(c.n_ids));
                        char * dst_base = reinterpret_cast<char *>(dst) + id * c.dst_nb1 + token * c.dst_nb2;
                        *reinterpret_cast<float *>(dst_base + row * sizeof(float)) =
                            sum[(wsic * TM + cr) * TN + cc];
                    }
                }
            }
        }
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32(
        const hrx_block_q4_K_moe_mmq * gate,
        const hrx_block_q4_K_moe_mmq * up,
        const hrx_block_q8_1_x4_moe_mmq * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    constexpr int BM = 16;
    constexpr int BN = 32;
    constexpr int BK_STEP = 1;
    constexpr int BLOCK_SIZE = 64;
    constexpr int WARP = 64;
    constexpr int WM = 16;
    constexpr int WN = 32;
    constexpr int WMITER = 1;
    constexpr int TM = 2;
    constexpr int TN = 2;
    constexpr int WNITER = (WM * WN) / (WARP * TM * TN * WMITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;
    constexpr int LOAD_VEC_A = 4;
    constexpr int LOAD_VEC_B = 16;
    constexpr int LOAD_STRIDE_A = BLOCK_SIZE * LOAD_VEC_A / 32;
    constexpr int LOAD_STRIDE_B = BLOCK_SIZE * LOAD_VEC_B / 32;
    constexpr int LOADS_A = BM / LOAD_STRIDE_A;
    constexpr int LOADS_B = BN / LOAD_STRIDE_B;

    static_assert(WNITER == 2, "unexpected Q4 MoE SWIGLU MMQ tile shape");
    static_assert(WSUBM == 16 && WSUBN == 16, "unexpected Q4 MoE SWIGLU MMQ subtile shape");
    static_assert(LOADS_A == 2 && LOADS_B == 1, "unexpected Q4 MoE SWIGLU MMQ load shape");

    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long row_base = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * BM;
    const uint32_t route_base0 = static_cast<uint32_t>(__builtin_amdgcn_workgroup_id_y()) * BN;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_z());
    if (expert >= c.n_experts) {
        return;
    }
    const uint32_t count = counts[expert];
    if (route_base0 >= count || row_base >= c.rows) {
        return;
    }

    const int tiw = static_cast<int>(tid);
    const int tiwr = tiw % (WSUBM / TM);
    const int tiwc = tiw / (WSUBM / TM);
    const uint32_t * expert_routes = routes + expert * c.route_capacity;
    const long long q8_blocks_per_col = c.k / 32;
    const uint32_t route_tile_span = static_cast<uint32_t>(((c.n_tokens + BN - 1) / BN) * BN);

    __shared__ hrx_q4_k_moe_a_cache gate_a[BM * BK_STEP];
    __shared__ hrx_q4_k_moe_a_cache up_a[BM * BK_STEP];
    __shared__ hrx_q8_1_moe_b_cache buf_b[BN * BK_STEP];

    for (uint32_t route_base = route_base0; route_base < count; route_base += route_tile_span) {
        float gate_sum[WNITER * TM * TN] = {};
        float up_sum[WNITER * TM * TN] = {};

        for (long long kb_base = 0; kb_base < q8_blocks_per_col; kb_base += BK_STEP) {
            const int loadr_a = static_cast<int>(tid % (32 / LOAD_VEC_A));
            const int loadc_a = static_cast<int>(tid / (32 / LOAD_VEC_A));
            const int loadr_b = static_cast<int>(tid % (32 / LOAD_VEC_B));
            const int loadc_b = static_cast<int>(tid / (32 / LOAD_VEC_B));
            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q4_k_moe_a_pending pending_gate[LOADS_A];
                hrx_q4_k_moe_a_pending pending_up[LOADS_A];
                hrx_q8_1_moe_b_pending pending_b[LOADS_B];
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_A; ++load_i) {
                    const int r = loadc_a + load_i * LOAD_STRIDE_A;
                    pending_gate[load_i] = hrx_q4_k_moe_mmq_fetch_a(
                        gate,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        c.rows,
                        c.gate_nb1,
                        c.gate_nb2,
                        expert);
                    pending_up[load_i] = hrx_q4_k_moe_mmq_fetch_a(
                        up,
                        row_base + r,
                        kb_base + k_step,
                        loadr_a,
                        c.rows,
                        c.up_nb1,
                        c.up_nb2,
                        expert);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    pending_b[load_i] = hrx_q8_1_moe_mmq_fetch_b(
                        src1,
                        expert_routes,
                        count,
                        route_base + col,
                        kb_base + k_step,
                        loadr_b,
                        q8_blocks_per_col);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_A; ++load_i) {
                    const int r = loadc_a + load_i * LOAD_STRIDE_A;
                    hrx_q4_k_moe_mmq_commit_a(gate_a, k_step * BM + r, pending_gate[load_i], loadr_a);
                    hrx_q4_k_moe_mmq_commit_a(up_a, k_step * BM + r, pending_up[load_i], loadr_a);
                }
                #pragma unroll
                for (int load_i = 0; load_i < LOADS_B; ++load_i) {
                    const int col = loadc_b + load_i * LOAD_STRIDE_B;
                    hrx_q8_1_moe_mmq_commit_b(buf_b, k_step * BN + col, pending_b[load_i], loadr_b);
                }
            }
            __syncthreads();

            #pragma unroll
            for (int k_step = 0; k_step < BK_STEP; ++k_step) {
                hrx_q4_k_moe_a_cache gate_cache[TM];
                hrx_q4_k_moe_a_cache up_cache[TM];
                #pragma unroll
                for (int cr = 0; cr < TM; ++cr) {
                    gate_cache[cr] = gate_a[k_step * BM + tiwr * TM + cr];
                    up_cache[cr] = up_a[k_step * BM + tiwr * TM + cr];
                }
                #pragma unroll
                for (int wsic = 0; wsic < WNITER; ++wsic) {
                    hrx_q8_1_moe_b_cache cache_b[TN];
                    #pragma unroll
                    for (int cc = 0; cc < TN; ++cc) {
                        cache_b[cc] = buf_b[k_step * BN + wsic * WSUBN + tiwc * TN + cc];
                    }
                    #pragma unroll
                    for (int cr = 0; cr < TM; ++cr) {
                        #pragma unroll
                        for (int cc = 0; cc < TN; ++cc) {
                            int gate_qsum = 0;
                            int up_qsum = 0;
                            #pragma unroll
                            for (int iqs = 0; iqs < 8; ++iqs) {
                                gate_qsum += hrx_udot4_q4_q8_1_moe_mmq(
                                    static_cast<uint32_t>(gate_cache[cr].qs[iqs]), cache_b[cc].qs[iqs]);
                                up_qsum += hrx_udot4_q4_q8_1_moe_mmq(
                                    static_cast<uint32_t>(up_cache[cr].qs[iqs]), cache_b[cc].qs[iqs]);
                            }
                            gate_sum[(wsic * TM + cr) * TN + cc] +=
                                gate_cache[cr].d * cache_b[cc].d * static_cast<float>(gate_qsum) -
                                gate_cache[cr].min * cache_b[cc].s;
                            up_sum[(wsic * TM + cr) * TN + cc] +=
                                up_cache[cr].d * cache_b[cc].d * static_cast<float>(up_qsum) -
                                up_cache[cr].min * cache_b[cc].s;
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
                const long long row = row_base + tiwr * TM + cr;
                #pragma unroll
                for (int cc = 0; cc < TN; ++cc) {
                    const int col = wsic * WSUBN + tiwc * TN + cc;
                    const uint32_t route_index = route_base + static_cast<uint32_t>(col);
                    if (row < c.rows && route_index < count) {
                        const uint32_t route = buf_b[col].route;
                        const long long id = static_cast<long long>(route % static_cast<uint32_t>(c.n_ids));
                        const long long token = static_cast<long long>(route / static_cast<uint32_t>(c.n_ids));
                        const int sum_idx = (wsic * TM + cr) * TN + cc;
                        const float gate_value = gate_sum[sum_idx];
                        const float silu_gate = gate_value / (1.0f + __expf(-gate_value));
                        char * dst_base = reinterpret_cast<char *>(dst) + id * c.dst_nb1 + token * c.dst_nb2;
                        *reinterpret_cast<float *>(dst_base + row * sizeof(float)) = up_sum[sum_idx] * silu_gate;
                    }
                }
            }
        }
    }
}
