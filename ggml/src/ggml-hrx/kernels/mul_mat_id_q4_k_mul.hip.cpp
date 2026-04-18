#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id_mul {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_mul_mat_id_q4_k_mul_constants {
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
    long long scale_nb1;
};

static __device__ __forceinline__ void hrx_get_scale_min_k4_id_mul(
        int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

template <int I>
static __device__ __forceinline__ float hrx_q4_k_mul_q_from_pack(unsigned long long pack, bool high) {
    const unsigned int byte = static_cast<unsigned int>((pack >> (8 * I)) & 0xFFu);
    return high ? static_cast<float>(byte >> 4) : static_cast<float>(byte & 0x0Fu);
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
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_mul_f32_impl(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
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
        const hrx_block_q4_K_id_mul * block = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id_mul));

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_id_mul(group, block->scales, &sc, &m);

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
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum * scale_value;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_mul_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    hrx_mul_mat_id_q4_k_mul_f32_impl<256>(src0, src1, ids, scale, dst, c);
}

static __global__ void hrx_mul_mat_id_q4_k_mul_wg128_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    hrx_mul_mat_id_q4_k_mul_f32_impl<128>(src0, src1, ids, scale, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_mul_wg64_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    hrx_mul_mat_id_q4_k_mul_f32_impl<64>(src0, src1, ids, scale, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_mul_packed_wg64_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
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

    __shared__ float sumsh[2];
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    const long long blocks_per_row = c.k / 256;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int il = itid >> 2;
    const int ir = itid - 4 * il;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_offset = 32 * v_im + l0;
    const int y_offset = 64 * v_im + l0;
    const int g0 = 2 * v_im;
    const int g1 = g0 + 1;
    const int g2 = g0 + 4;
    const int g3 = g2 + 1;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const hrx_block_q4_K_id_mul * block = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id_mul));

        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        uint8_t sc2 = 0;
        uint8_t m2 = 0;
        uint8_t sc3 = 0;
        uint8_t m3 = 0;
        hrx_get_scale_min_k4_id_mul(g0, block->scales, &sc0, &m0);
        hrx_get_scale_min_k4_id_mul(g1, block->scales, &sc1, &m1);
        hrx_get_scale_min_k4_id_mul(g2, block->scales, &sc2, &m2);
        hrx_get_scale_min_k4_id_mul(g3, block->scales, &sc3, &m3);

        const float d = __half2float(__ushort_as_half(block->d));
        const float dmin = __half2float(__ushort_as_half(block->dmin));
        const float d0 = d * static_cast<float>(sc0);
        const float d1 = d * static_cast<float>(sc1);
        const float d2 = d * static_cast<float>(sc2);
        const float d3 = d * static_cast<float>(sc3);
        const float min0 = dmin * static_cast<float>(m0);
        const float min1 = dmin * static_cast<float>(m1);
        const float min2 = dmin * static_cast<float>(m2);
        const float min3 = dmin * static_cast<float>(m3);

        const long long src_base = block_idx * 256 + y_offset;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t q01 = block->qs[q_offset + j];
            const uint8_t q23 = block->qs[q_offset + 64 + j];
            const float y0 = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const float y1 = *reinterpret_cast<const float *>(src1_col + (src_base + 32 + j) * sizeof(float));
            const float y2 = *reinterpret_cast<const float *>(src1_col + (src_base + 128 + j) * sizeof(float));
            const float y3 = *reinterpret_cast<const float *>(src1_col + (src_base + 160 + j) * sizeof(float));
            sum += (d0 * static_cast<float>(q01 & 0x0F) - min0) * y0;
            sum += (d1 * static_cast<float>(q01 >> 4) - min1) * y1;
            sum += (d2 * static_cast<float>(q23 & 0x0F) - min2) * y2;
            sum += (d3 * static_cast<float>(q23 >> 4) - min3) * y3;
        }
    }

    sum = hrx_reduce_wg<64>(sum, sumsh);

    if (tid == 0) {
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum * scale_value;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long row = row0 + static_cast<long long>(tid >> 4);
    const long long outer = __builtin_amdgcn_workgroup_id_y();
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

    const unsigned int lane = tid & 15;
    const char * src0_row_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2 + row * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 4; ++iter) {
        const long long block_idx = iter >> 1;
        const int group = ((iter & 1) << 2) + static_cast<int>(lane >> 2);
        const int group_offset = static_cast<int>(lane & 3) << 3;
        const long long col = static_cast<long long>(iter) * 128 + static_cast<long long>(lane) * 8;
        const hrx_block_q4_K_id_mul * block = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row_base + block_idx * sizeof(hrx_block_q4_K_id_mul));

        uint8_t sc = 0;
        uint8_t m = 0;
        hrx_get_scale_min_k4_id_mul(group, block->scales, &sc, &m);
        const float d = __half2float(__ushort_as_half(block->d)) * static_cast<float>(sc);
        const float min = __half2float(__ushort_as_half(block->dmin)) * static_cast<float>(m);
        const int qs_base = (group >> 1) * 32 + group_offset;

        const bool high = (group & 1) != 0;
        const unsigned long long qpack = *reinterpret_cast<const unsigned long long *>(block->qs + qs_base);
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col + col * sizeof(float));
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col + (col + 4) * sizeof(float));

        sum += (d * hrx_q4_k_mul_q_from_pack<0>(qpack, high) - min) * b0.x;
        sum += (d * hrx_q4_k_mul_q_from_pack<1>(qpack, high) - min) * b0.y;
        sum += (d * hrx_q4_k_mul_q_from_pack<2>(qpack, high) - min) * b0.z;
        sum += (d * hrx_q4_k_mul_q_from_pack<3>(qpack, high) - min) * b0.w;
        sum += (d * hrx_q4_k_mul_q_from_pack<4>(qpack, high) - min) * b1.x;
        sum += (d * hrx_q4_k_mul_q_from_pack<5>(qpack, high) - min) * b1.y;
        sum += (d * hrx_q4_k_mul_q_from_pack<6>(qpack, high) - min) * b1.z;
        sum += (d * hrx_q4_k_mul_q_from_pack<7>(qpack, high) - min) * b1.w;
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 16);
    }

    if (lane == 0) {
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum * scale_value;
    }
}

static __global__ void hrx_mul_mat_id_q4_k_mul_rows2_x16_wg16_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    if (row0 >= c.rows) {
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

    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_expert_base + row1 * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    const bool have_row1 = row1 < c.rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (long long col = static_cast<long long>(tid) * 8; col < c.k; col += 128) {
        const long long block_idx = col >> 8;
        const int group = static_cast<int>((col & 255) >> 5);
        const int group_offset = static_cast<int>(col & 31);
        const int qs_base = (group >> 1) * 32 + group_offset;
        const hrx_block_q4_K_id_mul * block0 = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id_mul));
        const hrx_block_q4_K_id_mul * block1 = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id_mul));

        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        hrx_get_scale_min_k4_id_mul(group, block0->scales, &sc0, &m0);
        const float d0 = __half2float(__ushort_as_half(block0->d)) * static_cast<float>(sc0);
        const float min0 = __half2float(__ushort_as_half(block0->dmin)) * static_cast<float>(m0);

        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        float d1 = 0.0f;
        float min1 = 0.0f;
        if (have_row1) {
            hrx_get_scale_min_k4_id_mul(group, block1->scales, &sc1, &m1);
            d1 = __half2float(__ushort_as_half(block1->d)) * static_cast<float>(sc1);
            min1 = __half2float(__ushort_as_half(block1->dmin)) * static_cast<float>(m1);
        }

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (col + j) * sizeof(float));
            const uint8_t packed0 = block0->qs[qs_base + j];
            const float q0 = (group & 1) ?
                static_cast<float>(packed0 >> 4) :
                static_cast<float>(packed0 & 0x0F);
            sum0 += (d0 * q0 - min0) * b;

            if (have_row1) {
                const uint8_t packed1 = block1->qs[qs_base + j];
                const float q1 = (group & 1) ?
                    static_cast<float>(packed1 >> 4) :
                    static_cast<float>(packed1 & 0x0F);
                sum1 += (d1 * q1 - min1) * b;
            }
        }
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 16);
        sum1 += __shfl_down(sum1, offset, 16);
    }

    if (tid == 0) {
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row0 * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum0 * scale_value;
        if (have_row1) {
            *reinterpret_cast<float *>(
                reinterpret_cast<char *>(dst) + row1 * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
                sum1 * scale_value;
        }
    }
}

static __global__ void hrx_mul_mat_id_q4_k_mul_packed_2row_wg64_f32(
        const hrx_block_q4_K_id_mul * src0,
        const float * src1,
        const int * ids,
        const float * scale,
        float * dst,
        hrx_mul_mat_id_q4_k_mul_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= c.rows) {
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

    __shared__ float sumsh[2];
    const char * src0_expert_base = reinterpret_cast<const char *>(src0) + expert * c.src0_nb2;
    const char * src0_row0_base = src0_expert_base + row0 * c.src0_nb1;
    const char * src0_row1_base = src0_expert_base + row1 * c.src0_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    const long long blocks_per_row = c.k / 256;
    const int itid = tid & 15;
    const int block_slot = tid >> 4;
    const int il = itid >> 2;
    const int ir = itid - 4 * il;
    const int v_im = il >> 1;
    const int v_in = il & 1;
    const int l0 = 4 * (2 * ir + v_in);
    const int q_offset = 32 * v_im + l0;
    const int y_offset = 64 * v_im + l0;
    const int g0 = 2 * v_im;
    const int g1 = g0 + 1;
    const int g2 = g0 + 4;
    const int g3 = g2 + 1;
    const bool have_row1 = row1 < c.rows;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += 4) {
        const hrx_block_q4_K_id_mul * block0 = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row0_base + block_idx * sizeof(hrx_block_q4_K_id_mul));
        const hrx_block_q4_K_id_mul * block1 = reinterpret_cast<const hrx_block_q4_K_id_mul *>(
            src0_row1_base + block_idx * sizeof(hrx_block_q4_K_id_mul));

        uint8_t sc0 = 0;
        uint8_t m0 = 0;
        uint8_t sc1 = 0;
        uint8_t m1 = 0;
        uint8_t sc2 = 0;
        uint8_t m2 = 0;
        uint8_t sc3 = 0;
        uint8_t m3 = 0;
        hrx_get_scale_min_k4_id_mul(g0, block0->scales, &sc0, &m0);
        hrx_get_scale_min_k4_id_mul(g1, block0->scales, &sc1, &m1);
        hrx_get_scale_min_k4_id_mul(g2, block0->scales, &sc2, &m2);
        hrx_get_scale_min_k4_id_mul(g3, block0->scales, &sc3, &m3);

        const float d = __half2float(__ushort_as_half(block0->d));
        const float dmin = __half2float(__ushort_as_half(block0->dmin));
        const float d0 = d * static_cast<float>(sc0);
        const float d1 = d * static_cast<float>(sc1);
        const float d2 = d * static_cast<float>(sc2);
        const float d3 = d * static_cast<float>(sc3);
        const float min0 = dmin * static_cast<float>(m0);
        const float min1 = dmin * static_cast<float>(m1);
        const float min2 = dmin * static_cast<float>(m2);
        const float min3 = dmin * static_cast<float>(m3);

        uint8_t sc10 = 0;
        uint8_t m10 = 0;
        uint8_t sc11 = 0;
        uint8_t m11 = 0;
        uint8_t sc12 = 0;
        uint8_t m12 = 0;
        uint8_t sc13 = 0;
        uint8_t m13 = 0;
        if (have_row1) {
            hrx_get_scale_min_k4_id_mul(g0, block1->scales, &sc10, &m10);
            hrx_get_scale_min_k4_id_mul(g1, block1->scales, &sc11, &m11);
            hrx_get_scale_min_k4_id_mul(g2, block1->scales, &sc12, &m12);
            hrx_get_scale_min_k4_id_mul(g3, block1->scales, &sc13, &m13);
        }

        const float d_row1 = have_row1 ? __half2float(__ushort_as_half(block1->d)) : 0.0f;
        const float dmin_row1 = have_row1 ? __half2float(__ushort_as_half(block1->dmin)) : 0.0f;
        const float d10 = d_row1 * static_cast<float>(sc10);
        const float d11 = d_row1 * static_cast<float>(sc11);
        const float d12 = d_row1 * static_cast<float>(sc12);
        const float d13 = d_row1 * static_cast<float>(sc13);
        const float min10 = dmin_row1 * static_cast<float>(m10);
        const float min11 = dmin_row1 * static_cast<float>(m11);
        const float min12 = dmin_row1 * static_cast<float>(m12);
        const float min13 = dmin_row1 * static_cast<float>(m13);

        const long long src_base = block_idx * 256 + y_offset;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t q01 = block0->qs[q_offset + j];
            const uint8_t q23 = block0->qs[q_offset + 64 + j];
            const float y0 = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const float y1 = *reinterpret_cast<const float *>(src1_col + (src_base + 32 + j) * sizeof(float));
            const float y2 = *reinterpret_cast<const float *>(src1_col + (src_base + 128 + j) * sizeof(float));
            const float y3 = *reinterpret_cast<const float *>(src1_col + (src_base + 160 + j) * sizeof(float));
            sum0 += (d0 * static_cast<float>(q01 & 0x0F) - min0) * y0;
            sum0 += (d1 * static_cast<float>(q01 >> 4) - min1) * y1;
            sum0 += (d2 * static_cast<float>(q23 & 0x0F) - min2) * y2;
            sum0 += (d3 * static_cast<float>(q23 >> 4) - min3) * y3;

            if (have_row1) {
                const uint8_t q101 = block1->qs[q_offset + j];
                const uint8_t q123 = block1->qs[q_offset + 64 + j];
                sum1 += (d10 * static_cast<float>(q101 & 0x0F) - min10) * y0;
                sum1 += (d11 * static_cast<float>(q101 >> 4) - min11) * y1;
                sum1 += (d12 * static_cast<float>(q123 & 0x0F) - min12) * y2;
                sum1 += (d13 * static_cast<float>(q123 >> 4) - min13) * y3;
            }
        }
    }

    sum0 = hrx_reduce_wg<64>(sum0, sumsh);
    __syncthreads();
    sum1 = hrx_reduce_wg<64>(sum1, sumsh);

    if (tid == 0) {
        const float scale_value = *reinterpret_cast<const float *>(
            reinterpret_cast<const char *>(scale) + id_pos * c.scale_nb1);
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row0 * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            sum0 * scale_value;
        if (have_row1) {
            *reinterpret_cast<float *>(
                reinterpret_cast<char *>(dst) + row1 * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
                sum1 * scale_value;
        }
    }
}
