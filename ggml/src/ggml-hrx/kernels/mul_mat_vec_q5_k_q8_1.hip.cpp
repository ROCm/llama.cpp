#include "mul_mat_vec_q5_k_q8_1_common.hip.inc"

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
