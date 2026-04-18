#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_mul_mat_id_q4_k_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long src0_nb1;
    long long src0_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
    long long dst_nb1;
    long long dst_nb2;
};

struct hrx_clear_u32_constants {
    long long n;
};

struct hrx_compact_moe_routes_constants {
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long route_capacity;
    long long ids_nb0;
    long long ids_nb1;
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

static __device__ __forceinline__ void hrx_get_scale_min_k4_id(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

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
static __device__ __forceinline__ void hrx_reduce_wg2(float & sum0, float & sum1, float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_f32_impl(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[WG_SIZE / 32];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q4_K_id * block = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id));

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_id(group, block->scales, &sc, &m);

        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t packed = block->qs[qs_base + j];
            const float q = (group & 1) ?
                static_cast<float>(packed >> 4) :
                static_cast<float>(packed & 0x0F);
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            sum += (d * q - min) * b;
        }
    }

    sum = hrx_reduce_wg<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    hrx_mul_mat_id_q4_k_f32_impl<256>(src0, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_wg128_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    hrx_mul_mat_id_q4_k_f32_impl<128>(src0, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_wg64_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    hrx_mul_mat_id_q4_k_f32_impl<64>(src0, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_row4_wg64_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[64 / 32];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const char * src0_row2_base = src0_row1_base + c.src0_nb1;
    const char * src0_row3_base = src0_row2_base + c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id * block0 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block1 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block2 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row2_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block3 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row3_base + block_idx * sizeof(hrx_block_q4_K_id));

        uint8_t sc0 = 0;
        uint8_t sc1 = 0;
        uint8_t sc2 = 0;
        uint8_t sc3 = 0;
        uint8_t m0 = 0;
        uint8_t m1 = 0;
        uint8_t m2 = 0;
        uint8_t m3 = 0;
        hrx_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
        hrx_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
        hrx_get_scale_min_k4_id(group, block2->scales, &sc2, &m2);
        hrx_get_scale_min_k4_id(group, block3->scales, &sc3, &m3);

        const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
        const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
        const float d2 = __half2float(__ushort_as_half(block2->d)) * static_cast<float>(sc2);
        const float d3 = __half2float(__ushort_as_half(block3->d)) * static_cast<float>(sc3);
        const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
        const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
        const float min2 = __half2float(__ushort_as_half(block2->dmin)) * static_cast<float>(m2);
        const float min3 = __half2float(__ushort_as_half(block3->dmin)) * static_cast<float>(m3);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const uint8_t packed0 = block0->qs[qs_base + j];
            const uint8_t packed1 = block1->qs[qs_base + j];
            const uint8_t packed2 = block2->qs[qs_base + j];
            const uint8_t packed3 = block3->qs[qs_base + j];
            const float q0 = (group & 1) ? static_cast<float>(packed0 >> 4) : static_cast<float>(packed0 & 0x0F);
            const float q1 = (group & 1) ? static_cast<float>(packed1 >> 4) : static_cast<float>(packed1 & 0x0F);
            const float q2 = (group & 1) ? static_cast<float>(packed2 >> 4) : static_cast<float>(packed2 & 0x0F);
            const float q3 = (group & 1) ? static_cast<float>(packed3 >> 4) : static_cast<float>(packed3 & 0x0F);
            sum0 += (d0 * q0 - min0) * b;
            sum1 += (d1 * q1 - min1) * b;
            sum2 += (d2 * q2 - min2) * b;
            sum3 += (d3 * q3 - min3) * b;
        }
    }

    sum0 = hrx_reduce_wg<64>(sum0, sumsh);
    __syncthreads();
    sum1 = hrx_reduce_wg<64>(sum1, sumsh);
    __syncthreads();
    sum2 = hrx_reduce_wg<64>(sum2, sumsh);
    __syncthreads();
    sum3 = hrx_reduce_wg<64>(sum3, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = sum0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = sum1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = sum2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = sum3;
    }
}

static __global__ void hrx_mul_mat_id_q4_k_row8_wg64_f32(
        const hrx_block_q4_K_id * src0, const float * src1, const int * ids, float * dst,
        hrx_mul_mat_id_q4_k_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 8;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 7 >= c.rows) {
        return;
    }

    const long long id_pos = outer % c.n_ids;
    const long long token = outer / c.n_ids;
    if (token >= c.n_tokens) {
        return;
    }

    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    __shared__ float sumsh[64 / 32];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const char * src0_row2_base = src0_row1_base + c.src0_nb1;
    const char * src0_row3_base = src0_row2_base + c.src0_nb1;
    const char * src0_row4_base = src0_row3_base + c.src0_nb1;
    const char * src0_row5_base = src0_row4_base + c.src0_nb1;
    const char * src0_row6_base = src0_row5_base + c.src0_nb1;
    const char * src0_row7_base = src0_row6_base + c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id * block0 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block1 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block2 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row2_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block3 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row3_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block4 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row4_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block5 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row5_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block6 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row6_base + block_idx * sizeof(hrx_block_q4_K_id));
        const hrx_block_q4_K_id * block7 = reinterpret_cast<const hrx_block_q4_K_id *>(
            src0_row7_base + block_idx * sizeof(hrx_block_q4_K_id));

        uint8_t sc0 = 0;
        uint8_t sc1 = 0;
        uint8_t sc2 = 0;
        uint8_t sc3 = 0;
        uint8_t sc4 = 0;
        uint8_t sc5 = 0;
        uint8_t sc6 = 0;
        uint8_t sc7 = 0;
        uint8_t m0 = 0;
        uint8_t m1 = 0;
        uint8_t m2 = 0;
        uint8_t m3 = 0;
        uint8_t m4 = 0;
        uint8_t m5 = 0;
        uint8_t m6 = 0;
        uint8_t m7 = 0;
        hrx_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
        hrx_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
        hrx_get_scale_min_k4_id(group, block2->scales, &sc2, &m2);
        hrx_get_scale_min_k4_id(group, block3->scales, &sc3, &m3);
        hrx_get_scale_min_k4_id(group, block4->scales, &sc4, &m4);
        hrx_get_scale_min_k4_id(group, block5->scales, &sc5, &m5);
        hrx_get_scale_min_k4_id(group, block6->scales, &sc6, &m6);
        hrx_get_scale_min_k4_id(group, block7->scales, &sc7, &m7);

        const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
        const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
        const float d2 = __half2float(__ushort_as_half(block2->d)) * static_cast<float>(sc2);
        const float d3 = __half2float(__ushort_as_half(block3->d)) * static_cast<float>(sc3);
        const float d4 = __half2float(__ushort_as_half(block4->d)) * static_cast<float>(sc4);
        const float d5 = __half2float(__ushort_as_half(block5->d)) * static_cast<float>(sc5);
        const float d6 = __half2float(__ushort_as_half(block6->d)) * static_cast<float>(sc6);
        const float d7 = __half2float(__ushort_as_half(block7->d)) * static_cast<float>(sc7);
        const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
        const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
        const float min2 = __half2float(__ushort_as_half(block2->dmin)) * static_cast<float>(m2);
        const float min3 = __half2float(__ushort_as_half(block3->dmin)) * static_cast<float>(m3);
        const float min4 = __half2float(__ushort_as_half(block4->dmin)) * static_cast<float>(m4);
        const float min5 = __half2float(__ushort_as_half(block5->dmin)) * static_cast<float>(m5);
        const float min6 = __half2float(__ushort_as_half(block6->dmin)) * static_cast<float>(m6);
        const float min7 = __half2float(__ushort_as_half(block7->dmin)) * static_cast<float>(m7);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
#define HRX_Q4K_ROW8_ACC(N) \
            do { \
                const uint8_t packed = block##N->qs[qs_base + j]; \
                const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F); \
                sum##N += (d##N * q - min##N) * b; \
            } while (0)
            HRX_Q4K_ROW8_ACC(0);
            HRX_Q4K_ROW8_ACC(1);
            HRX_Q4K_ROW8_ACC(2);
            HRX_Q4K_ROW8_ACC(3);
            HRX_Q4K_ROW8_ACC(4);
            HRX_Q4K_ROW8_ACC(5);
            HRX_Q4K_ROW8_ACC(6);
            HRX_Q4K_ROW8_ACC(7);
#undef HRX_Q4K_ROW8_ACC
        }
    }

    sum0 = hrx_reduce_wg<64>(sum0, sumsh);
    __syncthreads();
    sum1 = hrx_reduce_wg<64>(sum1, sumsh);
    __syncthreads();
    sum2 = hrx_reduce_wg<64>(sum2, sumsh);
    __syncthreads();
    sum3 = hrx_reduce_wg<64>(sum3, sumsh);
    __syncthreads();
    sum4 = hrx_reduce_wg<64>(sum4, sumsh);
    __syncthreads();
    sum5 = hrx_reduce_wg<64>(sum5, sumsh);
    __syncthreads();
    sum6 = hrx_reduce_wg<64>(sum6, sumsh);
    __syncthreads();
    sum7 = hrx_reduce_wg<64>(sum7, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = sum0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = sum1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = sum2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = sum3;
        *reinterpret_cast<float *>(dst_base + (row0 + 4) * sizeof(float)) = sum4;
        *reinterpret_cast<float *>(dst_base + (row0 + 5) * sizeof(float)) = sum5;
        *reinterpret_cast<float *>(dst_base + (row0 + 6) * sizeof(float)) = sum6;
        *reinterpret_cast<float *>(dst_base + (row0 + 7) * sizeof(float)) = sum7;
    }
}

extern "C" __global__ void hrx_clear_u32(uint32_t * dst, hrx_clear_u32_constants c) {
    const long long idx =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * __builtin_amdgcn_workgroup_size_x() +
        static_cast<long long>(__builtin_amdgcn_workitem_id_x());
    if (idx < c.n) {
        dst[idx] = 0;
    }
}

extern "C" __global__ void hrx_compact_moe_routes_i32(
        const int * ids,
        uint32_t * counts,
        uint32_t * routes,
        hrx_compact_moe_routes_constants c) {
    const long long idx =
        static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * __builtin_amdgcn_workgroup_size_x() +
        static_cast<long long>(__builtin_amdgcn_workitem_id_x());
    const long long n_routes = c.n_ids * c.n_tokens;
    if (idx >= n_routes) {
        return;
    }
    const long long id_pos = idx % c.n_ids;
    const long long token = idx / c.n_ids;
    const int expert = *reinterpret_cast<const int *>(
        reinterpret_cast<const char *>(ids) + id_pos * c.ids_nb0 + token * c.ids_nb1);
    if (expert < 0 || expert >= c.n_experts) {
        return;
    }

    const uint32_t slot = atomicAdd(&counts[expert], 1u);
    if (slot < static_cast<uint32_t>(c.route_capacity)) {
        routes[static_cast<long long>(expert) * c.route_capacity + slot] = static_cast<uint32_t>(idx);
    }
}

static __global__ void hrx_mul_mat_id_q4_k_grouped_row4_wg64_f32(
        const hrx_block_q4_K_id * src0,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh0[64 / 32];
    __shared__ float sumsh1[64 / 32];
    __shared__ float sumsh2[64 / 32];
    __shared__ float sumsh3[64 / 32];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const char * src0_row2_base = src0_row1_base + c.src0_nb1;
    const char * src0_row3_base = src0_row2_base + c.src0_nb1;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (uint32_t route_base = 0; route_base < count; route_base += 4) {
        const uint32_t route0 = expert_routes[route_base + 0];
        const uint32_t route1 = route_base + 1 < count ? expert_routes[route_base + 1] : route0;
        const uint32_t route2 = route_base + 2 < count ? expert_routes[route_base + 2] : route0;
        const uint32_t route3 = route_base + 3 < count ? expert_routes[route_base + 3] : route0;
        const long long id0 = route0 % c.n_ids;
        const long long id1 = route1 % c.n_ids;
        const long long id2 = route2 % c.n_ids;
        const long long id3 = route3 % c.n_ids;
        const long long token0 = route0 / c.n_ids;
        const long long token1 = route1 / c.n_ids;
        const long long token2 = route2 / c.n_ids;
        const long long token3 = route3 / c.n_ids;
        const char * src1_col0 = reinterpret_cast<const char *>(src1) + id0 * c.src1_nb1 + token0 * c.src1_nb2;
        const char * src1_col1 = reinterpret_cast<const char *>(src1) + id1 * c.src1_nb1 + token1 * c.src1_nb2;
        const char * src1_col2 = reinterpret_cast<const char *>(src1) + id2 * c.src1_nb1 + token2 * c.src1_nb2;
        const char * src1_col3 = reinterpret_cast<const char *>(src1) + id3 * c.src1_nb1 + token3 * c.src1_nb2;

        float s00 = 0.0f;
        float s01 = 0.0f;
        float s02 = 0.0f;
        float s03 = 0.0f;
        float s10 = 0.0f;
        float s11 = 0.0f;
        float s12 = 0.0f;
        float s13 = 0.0f;
        float s20 = 0.0f;
        float s21 = 0.0f;
        float s22 = 0.0f;
        float s23 = 0.0f;
        float s30 = 0.0f;
        float s31 = 0.0f;
        float s32 = 0.0f;
        float s33 = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id * block0 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id));
            const hrx_block_q4_K_id * block1 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id));
            const hrx_block_q4_K_id * block2 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row2_base + block_idx * sizeof(hrx_block_q4_K_id));
            const hrx_block_q4_K_id * block3 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row3_base + block_idx * sizeof(hrx_block_q4_K_id));
            uint8_t sc0 = 0, sc1 = 0, sc2 = 0, sc3 = 0;
            uint8_t m0 = 0, m1 = 0, m2 = 0, m3 = 0;
            hrx_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
            hrx_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
            hrx_get_scale_min_k4_id(group, block2->scales, &sc2, &m2);
            hrx_get_scale_min_k4_id(group, block3->scales, &sc3, &m3);
            const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
            const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
            const float d2 = __half2float(__ushort_as_half(block2->d)) * static_cast<float>(sc2);
            const float d3 = __half2float(__ushort_as_half(block3->d)) * static_cast<float>(sc3);
            const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
            const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
            const float min2 = __half2float(__ushort_as_half(block2->dmin)) * static_cast<float>(m2);
            const float min3 = __half2float(__ushort_as_half(block3->dmin)) * static_cast<float>(m3);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float b0 = *reinterpret_cast<const float *>(src1_col0 + (src_base + j) * sizeof(float));
                const float b1 = *reinterpret_cast<const float *>(src1_col1 + (src_base + j) * sizeof(float));
                const float b2 = *reinterpret_cast<const float *>(src1_col2 + (src_base + j) * sizeof(float));
                const float b3 = *reinterpret_cast<const float *>(src1_col3 + (src_base + j) * sizeof(float));
#define HRX_Q4K_GROUPED_ACC(N) \
                do { \
                    const uint8_t packed = block##N->qs[qs_base + j]; \
                    const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F); \
                    const float v = d##N * q - min##N; \
                    s##N##0 += v * b0; \
                    s##N##1 += v * b1; \
                    s##N##2 += v * b2; \
                    s##N##3 += v * b3; \
                } while (0)
                HRX_Q4K_GROUPED_ACC(0);
                HRX_Q4K_GROUPED_ACC(1);
                HRX_Q4K_GROUPED_ACC(2);
                HRX_Q4K_GROUPED_ACC(3);
#undef HRX_Q4K_GROUPED_ACC
            }
        }

#define HRX_Q4K_GROUPED_STORE(ROUTE, SUM, SHARED, ROW) \
        do { \
            float reduced = hrx_reduce_wg<64>((SUM), (SHARED)); \
            if (tid == 0) { \
                char * dst_base = reinterpret_cast<char *>(dst) + \
                    static_cast<long long>((ROUTE) % c.n_ids) * c.dst_nb1 + \
                    static_cast<long long>((ROUTE) / c.n_ids) * c.dst_nb2; \
                *reinterpret_cast<float *>(dst_base + (row0 + (ROW)) * sizeof(float)) = reduced; \
            } \
            __syncthreads(); \
        } while (0)
        HRX_Q4K_GROUPED_STORE(route0, s00, sumsh0, 0);
        HRX_Q4K_GROUPED_STORE(route0, s10, sumsh1, 1);
        HRX_Q4K_GROUPED_STORE(route0, s20, sumsh2, 2);
        HRX_Q4K_GROUPED_STORE(route0, s30, sumsh3, 3);
        if (route_base + 1 < count) {
            HRX_Q4K_GROUPED_STORE(route1, s01, sumsh0, 0);
            HRX_Q4K_GROUPED_STORE(route1, s11, sumsh1, 1);
            HRX_Q4K_GROUPED_STORE(route1, s21, sumsh2, 2);
            HRX_Q4K_GROUPED_STORE(route1, s31, sumsh3, 3);
        }
        if (route_base + 2 < count) {
            HRX_Q4K_GROUPED_STORE(route2, s02, sumsh0, 0);
            HRX_Q4K_GROUPED_STORE(route2, s12, sumsh1, 1);
            HRX_Q4K_GROUPED_STORE(route2, s22, sumsh2, 2);
            HRX_Q4K_GROUPED_STORE(route2, s32, sumsh3, 3);
        }
        if (route_base + 3 < count) {
            HRX_Q4K_GROUPED_STORE(route3, s03, sumsh0, 0);
            HRX_Q4K_GROUPED_STORE(route3, s13, sumsh1, 1);
            HRX_Q4K_GROUPED_STORE(route3, s23, sumsh2, 2);
            HRX_Q4K_GROUPED_STORE(route3, s33, sumsh3, 3);
        }
#undef HRX_Q4K_GROUPED_STORE
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32(
        const hrx_block_q4_K_id * src0,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long expert = static_cast<long long>(__builtin_amdgcn_workgroup_id_y());
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh[8 * 2 * (64 / 32)];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_row0_base + c.src0_nb1;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (uint32_t route_base = 0; route_base < count; route_base += 8) {
        const uint32_t route_a = expert_routes[route_base];
        const uint32_t route_b = route_base + 1 < count ? expert_routes[route_base + 1] : route_a;
        const uint32_t route_c = route_base + 2 < count ? expert_routes[route_base + 2] : route_a;
        const uint32_t route_d = route_base + 3 < count ? expert_routes[route_base + 3] : route_a;
        const uint32_t route_e = route_base + 4 < count ? expert_routes[route_base + 4] : route_a;
        const uint32_t route_f = route_base + 5 < count ? expert_routes[route_base + 5] : route_a;
        const uint32_t route_g = route_base + 6 < count ? expert_routes[route_base + 6] : route_a;
        const uint32_t route_h = route_base + 7 < count ? expert_routes[route_base + 7] : route_a;
        const bool has_b = route_base + 1 < count;
        const bool has_c = route_base + 2 < count;
        const bool has_d = route_base + 3 < count;
        const bool has_e = route_base + 4 < count;
        const bool has_f = route_base + 5 < count;
        const bool has_g = route_base + 6 < count;
        const bool has_h = route_base + 7 < count;

#define HRX_Q4K_GROUPED_ROUTE_COL(S) \
        const long long id_##S = static_cast<long long>(route_##S & 7u); \
        const long long tok_##S = static_cast<long long>(route_##S >> 3); \
        const char * src1_##S = reinterpret_cast<const char *>(src1) + id_##S * c.src1_nb1 + tok_##S * c.src1_nb2
        HRX_Q4K_GROUPED_ROUTE_COL(a);
        HRX_Q4K_GROUPED_ROUTE_COL(b);
        HRX_Q4K_GROUPED_ROUTE_COL(c);
        HRX_Q4K_GROUPED_ROUTE_COL(d);
        HRX_Q4K_GROUPED_ROUTE_COL(e);
        HRX_Q4K_GROUPED_ROUTE_COL(f);
        HRX_Q4K_GROUPED_ROUTE_COL(g);
        HRX_Q4K_GROUPED_ROUTE_COL(h);
#undef HRX_Q4K_GROUPED_ROUTE_COL

        float s0a = 0.0f, s1a = 0.0f;
        float s0b = 0.0f, s1b = 0.0f;
        float s0c = 0.0f, s1c = 0.0f;
        float s0d = 0.0f, s1d = 0.0f;
        float s0e = 0.0f, s1e = 0.0f;
        float s0f = 0.0f, s1f = 0.0f;
        float s0g = 0.0f, s1g = 0.0f;
        float s0h = 0.0f, s1h = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id * block0 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id));
            const hrx_block_q4_K_id * block1 = reinterpret_cast<const hrx_block_q4_K_id *>(
                src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id));
            uint8_t sc0 = 0, sc1 = 0;
            uint8_t m0 = 0, m1 = 0;
            hrx_get_scale_min_k4_id(group, block0->scales, &sc0, &m0);
            hrx_get_scale_min_k4_id(group, block1->scales, &sc1, &m1);
            const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
            const float d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
            const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);
            const float min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;
            const uint32_t packed_qs0 = *reinterpret_cast<const uint32_t *>(block0->qs + qs_base);
            const uint32_t packed_qs1 = *reinterpret_cast<const uint32_t *>(block1->qs + qs_base);

#define HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(S) \
            const float4 b4_##S = *reinterpret_cast<const float4 *>(src1_##S + src_base * sizeof(float))
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(a);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(b);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(c);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(d);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(e);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(f);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(g);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4(h);
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD4

#define HRX_Q4K_GROUPED_ROW2_ROUTE8_ACC(N, J) \
                do { \
                    const uint8_t packed = static_cast<uint8_t>(packed_qs##N >> ((J) * 8)); \
                    const float q = (group & 1) ? static_cast<float>(packed >> 4) : static_cast<float>(packed & 0x0F); \
                    const float v = d##N * q - min##N; \
                    s##N##a += v * ba; s##N##b += v * bb; s##N##c += v * bc; s##N##d += v * bd; \
                    s##N##e += v * be; s##N##f += v * bf; s##N##g += v * bg; s##N##h += v * bh; \
                } while (0)
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP(J, FIELD) \
            do { \
                const float ba = b4_a.FIELD; \
                const float bb = b4_b.FIELD; \
                const float bc = b4_c.FIELD; \
                const float bd = b4_d.FIELD; \
                const float be = b4_e.FIELD; \
                const float bf = b4_f.FIELD; \
                const float bg = b4_g.FIELD; \
                const float bh = b4_h.FIELD; \
                HRX_Q4K_GROUPED_ROW2_ROUTE8_ACC(0, J); \
                HRX_Q4K_GROUPED_ROW2_ROUTE8_ACC(1, J); \
            } while (0)
            HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP(0, x);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP(1, y);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP(2, z);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP(3, w);
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_ACC
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_STEP
        }

        const unsigned int reduce_lane = tid & (warpSize - 1);
        const unsigned int wave = tid / warpSize;
        constexpr int waves = 64 / 32;
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(S) \
        do { \
            s0##S += __shfl_down(s0##S, offset); \
            s1##S += __shfl_down(s1##S, offset); \
        } while (0)
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(a);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(b);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(c);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(d);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(e);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(f);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(g);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(h);
        }
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(S, R) \
        do { \
            sumsh[wave + ((R) * 2 + 0) * waves] = s0##S; \
            sumsh[wave + ((R) * 2 + 1) * waves] = s1##S; \
        } while (0)
        if (reduce_lane == 0) {
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(a, 0);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(b, 1);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(c, 2);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(d, 3);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(e, 4);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(f, 5);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(g, 6);
            HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE(h, 7);
        }
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_SAVE
        __syncthreads();
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(S, R) \
        do { \
            s0##S = reduce_lane < waves ? sumsh[reduce_lane + ((R) * 2 + 0) * waves] : 0.0f; \
            s1##S = reduce_lane < waves ? sumsh[reduce_lane + ((R) * 2 + 1) * waves] : 0.0f; \
        } while (0)
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(a, 0);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(b, 1);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(c, 2);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(d, 3);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(e, 4);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(f, 5);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(g, 6);
        HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD(h, 7);
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_LOAD
        if (wave == 0) {
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(S) \
            do { \
                s0##S += __shfl_down(s0##S, offset); \
                s1##S += __shfl_down(s1##S, offset); \
            } while (0)
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(a);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(b);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(c);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(d);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(e);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(f);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(g);
                HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL(h);
            }
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_SHFL
        }
#define HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(S, ROUTE) \
        do { \
            if (tid == 0) { \
                const uint32_t route = (ROUTE); \
                const long long id = static_cast<long long>(route & 7u); \
                const long long token = static_cast<long long>(route >> 3); \
                char * dst_base = reinterpret_cast<char *>(dst) + id * c.dst_nb1 + token * c.dst_nb2; \
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = s0##S; \
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = s1##S; \
            } \
        } while (0)
        HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(a, route_a);
        if (has_b) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(b, route_b); }
        if (has_c) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(c, route_c); }
        if (has_d) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(d, route_d); }
        if (has_e) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(e, route_e); }
        if (has_f) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(f, route_f); }
        if (has_g) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(g, route_g); }
        if (has_h) { HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE(h, route_h); }
#undef HRX_Q4K_GROUPED_ROW2_ROUTE8_STORE
        __syncthreads();
    }
}
