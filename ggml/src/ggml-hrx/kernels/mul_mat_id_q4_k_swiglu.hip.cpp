#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_block_q4_K_id_swiglu {
    unsigned short d;
    unsigned short dmin;
    uint8_t scales[12];
    uint8_t qs[128];
};

struct hrx_mul_mat_id_q4_k_swiglu_constants {
    long long k;
    long long rows;
    long long n_ids;
    long long n_tokens;
    long long n_experts;
    long long gate_nb1;
    long long gate_nb2;
    long long up_nb1;
    long long up_nb2;
    long long src1_nb1;
    long long src1_nb2;
    long long ids_nb0;
    long long ids_nb1;
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

static __device__ __forceinline__ void hrx_get_scale_min_k4_id_swiglu(
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
static __device__ __forceinline__ float hrx_q4_k_swiglu_q_from_pack(unsigned long long pack, bool high) {
    const unsigned int byte = static_cast<unsigned int>((pack >> (8 * I)) & 0xFFu);
    return high ? static_cast<float>(byte >> 4) : static_cast<float>(byte & 0x0Fu);
}

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_wg_swiglu(float sum, float * shared) {
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
static __device__ __forceinline__ void hrx_reduce_wg4_swiglu(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    constexpr int waves = (WG_SIZE + 31) / 32;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_id_q4_k_swiglu_f32_impl(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
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

    __shared__ float gate_sumsh[WG_SIZE / 32];
    __shared__ float up_sumsh[WG_SIZE / 32];
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    const int block_lane = tid & 63;
    const int block_slot = tid >> 6;
    const int block_stride = WG_SIZE >> 6;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = block_slot; block_idx < blocks_per_row; block_idx += block_stride) {
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc = 0;
        uint8_t gate_m = 0;
        uint8_t up_sc = 0;
        uint8_t up_m = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block->scales, &gate_sc, &gate_m);
        hrx_get_scale_min_k4_id_swiglu(group, up_block->scales, &up_sc, &up_m);

        const float gate_d = __half2float(__ushort_as_half(gate_block->d)) * static_cast<float>(gate_sc);
        const float gate_min = __half2float(__ushort_as_half(gate_block->dmin)) * static_cast<float>(gate_m);
        const float up_d = __half2float(__ushort_as_half(up_block->d)) * static_cast<float>(up_sc);
        const float up_min = __half2float(__ushort_as_half(up_block->dmin)) * static_cast<float>(up_m);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint8_t gate_packed = gate_block->qs[qs_base + j];
            const uint8_t up_packed = up_block->qs[qs_base + j];
            const float gate_q = (group & 1) ?
                static_cast<float>(gate_packed >> 4) :
                static_cast<float>(gate_packed & 0x0F);
            const float up_q = (group & 1) ?
                static_cast<float>(up_packed >> 4) :
                static_cast<float>(up_packed & 0x0F);
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            gate_sum += (gate_d * gate_q - gate_min) * b;
            up_sum += (up_d * up_q - up_min) * b;
        }
    }

    gate_sum = hrx_reduce_wg_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_wg_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<256>(gate, up, src1, ids, dst, c);
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_wg128_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<128>(gate, up, src1, ids, dst, c);
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    hrx_mul_mat_id_q4_k_swiglu_f32_impl<64>(gate, up, src1, ids, dst, c);
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_row2_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long outer = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows) {
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
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
        hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
        hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

        const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
        const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
        const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
        const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
        const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
        const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
        const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
        const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
            const uint8_t gate_packed0 = gate_block0->qs[qs_base + j];
            const uint8_t gate_packed1 = gate_block1->qs[qs_base + j];
            const uint8_t up_packed0 = up_block0->qs[qs_base + j];
            const uint8_t up_packed1 = up_block1->qs[qs_base + j];
            const float gate_q0 = (group & 1) ?
                static_cast<float>(gate_packed0 >> 4) :
                static_cast<float>(gate_packed0 & 0x0F);
            const float gate_q1 = (group & 1) ?
                static_cast<float>(gate_packed1 >> 4) :
                static_cast<float>(gate_packed1 & 0x0F);
            const float up_q0 = (group & 1) ?
                static_cast<float>(up_packed0 >> 4) :
                static_cast<float>(up_packed0 & 0x0F);
            const float up_q1 = (group & 1) ?
                static_cast<float>(up_packed1 >> 4) :
                static_cast<float>(up_packed1 & 0x0F);
            gate_sum0 += (gate_d0 * gate_q0 - gate_min0) * b;
            gate_sum1 += (gate_d1 * gate_q1 - gate_min1) * b;
            up_sum0 += (up_d0 * up_q0 - up_min0) * b;
            up_sum1 += (up_d1 * up_q1 - up_min1) * b;
        }
    }

    gate_sum0 = hrx_reduce_wg_swiglu<64>(gate_sum0, sumsh);
    __syncthreads();
    up_sum0 = hrx_reduce_wg_swiglu<64>(up_sum0, sumsh);
    __syncthreads();
    gate_sum1 = hrx_reduce_wg_swiglu<64>(gate_sum1, sumsh);
    __syncthreads();
    up_sum1 = hrx_reduce_wg_swiglu<64>(up_sum1, sumsh);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0 * silu_gate0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1 * silu_gate1;
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
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

    __shared__ float gate_sumsh0[64 / 32];
    __shared__ float up_sumsh0[64 / 32];
    __shared__ float gate_sumsh1[64 / 32];
    __shared__ float up_sumsh1[64 / 32];
    __shared__ float gate_sumsh2[64 / 32];
    __shared__ float up_sumsh2[64 / 32];
    __shared__ float gate_sumsh3[64 / 32];
    __shared__ float up_sumsh3[64 / 32];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * gate_row2_base = gate_row1_base + c.gate_nb1;
    const char * gate_row3_base = gate_row2_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * up_row2_base = up_row1_base + c.up_nb1;
    const char * up_row3_base = up_row2_base + c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum0 = 0.0f;
    float up_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float up_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float up_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float up_sum3 = 0.0f;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;

    for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * gate_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t gate_sc2 = 0;
        uint8_t gate_m2 = 0;
        uint8_t gate_sc3 = 0;
        uint8_t gate_m3 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        uint8_t up_sc2 = 0;
        uint8_t up_m2 = 0;
        uint8_t up_sc3 = 0;
        uint8_t up_m3 = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block2->scales, &gate_sc2, &gate_m2);
        hrx_get_scale_min_k4_id_swiglu(group, gate_block3->scales, &gate_sc3, &gate_m3);
        hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
        hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);
        hrx_get_scale_min_k4_id_swiglu(group, up_block2->scales, &up_sc2, &up_m2);
        hrx_get_scale_min_k4_id_swiglu(group, up_block3->scales, &up_sc3, &up_m3);

        const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
        const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
        const float gate_d2 = __half2float(__ushort_as_half(gate_block2->d)) * static_cast<float>(gate_sc2);
        const float gate_d3 = __half2float(__ushort_as_half(gate_block3->d)) * static_cast<float>(gate_sc3);
        const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
        const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
        const float up_d2 = __half2float(__ushort_as_half(up_block2->d)) * static_cast<float>(up_sc2);
        const float up_d3 = __half2float(__ushort_as_half(up_block3->d)) * static_cast<float>(up_sc3);
        const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
        const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
        const float gate_min2 = __half2float(__ushort_as_half(gate_block2->dmin)) * static_cast<float>(gate_m2);
        const float gate_min3 = __half2float(__ushort_as_half(gate_block3->dmin)) * static_cast<float>(gate_m3);
        const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
        const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
        const float up_min2 = __half2float(__ushort_as_half(up_block2->dmin)) * static_cast<float>(up_m2);
        const float up_min3 = __half2float(__ushort_as_half(up_block3->dmin)) * static_cast<float>(up_m3);
        const long long src_base = block_idx * 256 + group * 32 + lane;
        const int qs_base = (group >> 1) * 32 + lane;

        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float b = *reinterpret_cast<const float *>(src1_col + (src_base + j) * sizeof(float));
#define HRX_Q4K_SWIGLU_ROW4_ACC(N) \
            do { \
                const uint8_t gate_packed = gate_block##N->qs[qs_base + j]; \
                const uint8_t up_packed = up_block##N->qs[qs_base + j]; \
                const float gate_q = (group & 1) ? static_cast<float>(gate_packed >> 4) : static_cast<float>(gate_packed & 0x0F); \
                const float up_q = (group & 1) ? static_cast<float>(up_packed >> 4) : static_cast<float>(up_packed & 0x0F); \
                gate_sum##N += (gate_d##N * gate_q - gate_min##N) * b; \
                up_sum##N += (up_d##N * up_q - up_min##N) * b; \
            } while (0)
            HRX_Q4K_SWIGLU_ROW4_ACC(0);
            HRX_Q4K_SWIGLU_ROW4_ACC(1);
            HRX_Q4K_SWIGLU_ROW4_ACC(2);
            HRX_Q4K_SWIGLU_ROW4_ACC(3);
#undef HRX_Q4K_SWIGLU_ROW4_ACC
        }
    }

    gate_sum0 = hrx_reduce_wg_swiglu<64>(gate_sum0, gate_sumsh0);
    up_sum0 = hrx_reduce_wg_swiglu<64>(up_sum0, up_sumsh0);
    gate_sum1 = hrx_reduce_wg_swiglu<64>(gate_sum1, gate_sumsh1);
    up_sum1 = hrx_reduce_wg_swiglu<64>(up_sum1, up_sumsh1);
    gate_sum2 = hrx_reduce_wg_swiglu<64>(gate_sum2, gate_sumsh2);
    up_sum2 = hrx_reduce_wg_swiglu<64>(up_sum2, up_sumsh2);
    gate_sum3 = hrx_reduce_wg_swiglu<64>(gate_sum3, gate_sumsh3);
    up_sum3 = hrx_reduce_wg_swiglu<64>(up_sum3, up_sumsh3);

    if (tid == 0) {
        char * dst_base = reinterpret_cast<char *>(dst) + id_pos * c.dst_nb1 + token * c.dst_nb2;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0 * silu_gate0;
        *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1 * silu_gate1;
        *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2 * silu_gate2;
        *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3 * silu_gate3;
    }
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 3 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float gate_sumsh0[64 / 32];
    __shared__ float up_sumsh0[64 / 32];
    __shared__ float gate_sumsh1[64 / 32];
    __shared__ float up_sumsh1[64 / 32];
    __shared__ float gate_sumsh2[64 / 32];
    __shared__ float up_sumsh2[64 / 32];
    __shared__ float gate_sumsh3[64 / 32];
    __shared__ float up_sumsh3[64 / 32];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * gate_row2_base = gate_row1_base + c.gate_nb1;
    const char * gate_row3_base = gate_row2_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;
    const char * up_row2_base = up_row1_base + c.up_nb1;
    const char * up_row3_base = up_row2_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 2) {
        const uint32_t route0 = expert_routes[route_idx];
        const uint32_t route1 = route_idx + 1 < count ? expert_routes[route_idx + 1] : route0;
        const bool has_route1 = route_idx + 1 < count;
        const long long id_pos0 = route0 % c.n_ids;
        const long long token0 = route0 / c.n_ids;
        const long long id_pos1 = route1 % c.n_ids;
        const long long token1 = route1 / c.n_ids;

        const char * src1_col0 = reinterpret_cast<const char *>(src1) + id_pos0 * c.src1_nb1 + token0 * c.src1_nb2;
        const char * src1_col1 = reinterpret_cast<const char *>(src1) + id_pos1 * c.src1_nb1 + token1 * c.src1_nb2;
        float gate_sum0a = 0.0f;
        float up_sum0a = 0.0f;
        float gate_sum1a = 0.0f;
        float up_sum1a = 0.0f;
        float gate_sum2a = 0.0f;
        float up_sum2a = 0.0f;
        float gate_sum3a = 0.0f;
        float up_sum3a = 0.0f;
        float gate_sum0b = 0.0f;
        float up_sum0b = 0.0f;
        float gate_sum1b = 0.0f;
        float up_sum1b = 0.0f;
        float gate_sum2b = 0.0f;
        float up_sum2b = 0.0f;
        float gate_sum3b = 0.0f;
        float up_sum3b = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block2 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row2_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block3 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row3_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0;
            uint8_t gate_m0 = 0;
            uint8_t gate_sc1 = 0;
            uint8_t gate_m1 = 0;
            uint8_t gate_sc2 = 0;
            uint8_t gate_m2 = 0;
            uint8_t gate_sc3 = 0;
            uint8_t gate_m3 = 0;
            uint8_t up_sc0 = 0;
            uint8_t up_m0 = 0;
            uint8_t up_sc1 = 0;
            uint8_t up_m1 = 0;
            uint8_t up_sc2 = 0;
            uint8_t up_m2 = 0;
            uint8_t up_sc3 = 0;
            uint8_t up_m3 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block2->scales, &gate_sc2, &gate_m2);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block3->scales, &gate_sc3, &gate_m3);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block2->scales, &up_sc2, &up_m2);
            hrx_get_scale_min_k4_id_swiglu(group, up_block3->scales, &up_sc3, &up_m3);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float gate_d2 = __half2float(__ushort_as_half(gate_block2->d)) * static_cast<float>(gate_sc2);
            const float gate_d3 = __half2float(__ushort_as_half(gate_block3->d)) * static_cast<float>(gate_sc3);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float up_d2 = __half2float(__ushort_as_half(up_block2->d)) * static_cast<float>(up_sc2);
            const float up_d3 = __half2float(__ushort_as_half(up_block3->d)) * static_cast<float>(up_sc3);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float gate_min2 = __half2float(__ushort_as_half(gate_block2->dmin)) * static_cast<float>(gate_m2);
            const float gate_min3 = __half2float(__ushort_as_half(gate_block3->dmin)) * static_cast<float>(gate_m3);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const float up_min2 = __half2float(__ushort_as_half(up_block2->dmin)) * static_cast<float>(up_m2);
            const float up_min3 = __half2float(__ushort_as_half(up_block3->dmin)) * static_cast<float>(up_m3);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float b0 = *reinterpret_cast<const float *>(src1_col0 + (src_base + j) * sizeof(float));
                const float b1 = has_route1 ?
                    *reinterpret_cast<const float *>(src1_col1 + (src_base + j) * sizeof(float)) : 0.0f;
#define HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(N) \
                do { \
                    const uint8_t gate_packed = gate_block##N->qs[qs_base + j]; \
                    const uint8_t up_packed = up_block##N->qs[qs_base + j]; \
                    const float gate_q = (group & 1) ? static_cast<float>(gate_packed >> 4) : static_cast<float>(gate_packed & 0x0F); \
                    const float up_q = (group & 1) ? static_cast<float>(up_packed >> 4) : static_cast<float>(up_packed & 0x0F); \
                    const float gate_val = gate_d##N * gate_q - gate_min##N; \
                    const float up_val = up_d##N * up_q - up_min##N; \
                    gate_sum##N##a += gate_val * b0; \
                    up_sum##N##a += up_val * b0; \
                    gate_sum##N##b += gate_val * b1; \
                    up_sum##N##b += up_val * b1; \
                } while (0)
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(0);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(1);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(2);
                HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC(3);
#undef HRX_Q4K_SWIGLU_GROUPED_ROW4_ACC
            }
        }

        gate_sum0a = hrx_reduce_wg_swiglu<64>(gate_sum0a, gate_sumsh0);
        up_sum0a = hrx_reduce_wg_swiglu<64>(up_sum0a, up_sumsh0);
        gate_sum1a = hrx_reduce_wg_swiglu<64>(gate_sum1a, gate_sumsh1);
        up_sum1a = hrx_reduce_wg_swiglu<64>(up_sum1a, up_sumsh1);
        gate_sum2a = hrx_reduce_wg_swiglu<64>(gate_sum2a, gate_sumsh2);
        up_sum2a = hrx_reduce_wg_swiglu<64>(up_sum2a, up_sumsh2);
        gate_sum3a = hrx_reduce_wg_swiglu<64>(gate_sum3a, gate_sumsh3);
        up_sum3a = hrx_reduce_wg_swiglu<64>(up_sum3a, up_sumsh3);

        if (tid == 0) {
            char * dst_base = reinterpret_cast<char *>(dst) + id_pos0 * c.dst_nb1 + token0 * c.dst_nb2;
            const float silu_gate0 = gate_sum0a / (1.0f + __expf(-gate_sum0a));
            const float silu_gate1 = gate_sum1a / (1.0f + __expf(-gate_sum1a));
            const float silu_gate2 = gate_sum2a / (1.0f + __expf(-gate_sum2a));
            const float silu_gate3 = gate_sum3a / (1.0f + __expf(-gate_sum3a));
            *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0a * silu_gate0;
            *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1a * silu_gate1;
            *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2a * silu_gate2;
            *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3a * silu_gate3;
        }
        __syncthreads();

        if (has_route1) {
            gate_sum0b = hrx_reduce_wg_swiglu<64>(gate_sum0b, gate_sumsh0);
            up_sum0b = hrx_reduce_wg_swiglu<64>(up_sum0b, up_sumsh0);
            gate_sum1b = hrx_reduce_wg_swiglu<64>(gate_sum1b, gate_sumsh1);
            up_sum1b = hrx_reduce_wg_swiglu<64>(up_sum1b, up_sumsh1);
            gate_sum2b = hrx_reduce_wg_swiglu<64>(gate_sum2b, gate_sumsh2);
            up_sum2b = hrx_reduce_wg_swiglu<64>(up_sum2b, up_sumsh2);
            gate_sum3b = hrx_reduce_wg_swiglu<64>(gate_sum3b, gate_sumsh3);
            up_sum3b = hrx_reduce_wg_swiglu<64>(up_sum3b, up_sumsh3);

            if (tid == 0) {
                char * dst_base = reinterpret_cast<char *>(dst) + id_pos1 * c.dst_nb1 + token1 * c.dst_nb2;
                const float silu_gate0 = gate_sum0b / (1.0f + __expf(-gate_sum0b));
                const float silu_gate1 = gate_sum1b / (1.0f + __expf(-gate_sum1b));
                const float silu_gate2 = gate_sum2b / (1.0f + __expf(-gate_sum2b));
                const float silu_gate3 = gate_sum3b / (1.0f + __expf(-gate_sum3b));
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0b * silu_gate0;
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1b * silu_gate1;
                *reinterpret_cast<float *>(dst_base + (row0 + 2) * sizeof(float)) = up_sum2b * silu_gate2;
                *reinterpret_cast<float *>(dst_base + (row0 + 3) * sizeof(float)) = up_sum3b * silu_gate3;
            }
        }
        __syncthreads();
    }
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route4_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh[8 * 4 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 4) {
        const uint32_t route_a = expert_routes[route_idx];
        const uint32_t route_b = route_idx + 1 < count ? expert_routes[route_idx + 1] : route_a;
        const uint32_t route_c = route_idx + 2 < count ? expert_routes[route_idx + 2] : route_a;
        const uint32_t route_d = route_idx + 3 < count ? expert_routes[route_idx + 3] : route_a;
        const bool has_b = route_idx + 1 < count;
        const bool has_c = route_idx + 2 < count;
        const bool has_d = route_idx + 3 < count;
        const long long id_a = route_a % c.n_ids;
        const long long tok_a = route_a / c.n_ids;
        const long long id_b = route_b % c.n_ids;
        const long long tok_b = route_b / c.n_ids;
        const long long id_c = route_c % c.n_ids;
        const long long tok_c = route_c / c.n_ids;
        const long long id_d = route_d % c.n_ids;
        const long long tok_d = route_d / c.n_ids;
        const char * src1_a = reinterpret_cast<const char *>(src1) + id_a * c.src1_nb1 + tok_a * c.src1_nb2;
        const char * src1_b = reinterpret_cast<const char *>(src1) + id_b * c.src1_nb1 + tok_b * c.src1_nb2;
        const char * src1_c = reinterpret_cast<const char *>(src1) + id_c * c.src1_nb1 + tok_c * c.src1_nb2;
        const char * src1_d = reinterpret_cast<const char *>(src1) + id_d * c.src1_nb1 + tok_d * c.src1_nb2;

        float gate_sum0a = 0.0f, up_sum0a = 0.0f, gate_sum1a = 0.0f, up_sum1a = 0.0f;
        float gate_sum0b = 0.0f, up_sum0b = 0.0f, gate_sum1b = 0.0f, up_sum1b = 0.0f;
        float gate_sum0c = 0.0f, up_sum0c = 0.0f, gate_sum1c = 0.0f, up_sum1c = 0.0f;
        float gate_sum0d = 0.0f, up_sum0d = 0.0f, gate_sum1d = 0.0f, up_sum1d = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0, gate_m0 = 0, gate_sc1 = 0, gate_m1 = 0;
            uint8_t up_sc0 = 0, up_m0 = 0, up_sc1 = 0, up_m1 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float ba = *reinterpret_cast<const float *>(src1_a + (src_base + j) * sizeof(float));
                const float bb = has_b ? *reinterpret_cast<const float *>(src1_b + (src_base + j) * sizeof(float)) : 0.0f;
                const float bc = has_c ? *reinterpret_cast<const float *>(src1_c + (src_base + j) * sizeof(float)) : 0.0f;
                const float bd = has_d ? *reinterpret_cast<const float *>(src1_d + (src_base + j) * sizeof(float)) : 0.0f;
#define HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(N) \
                do { \
                    const uint8_t gate_packed = gate_block##N->qs[qs_base + j]; \
                    const uint8_t up_packed = up_block##N->qs[qs_base + j]; \
                    const float gate_q = (group & 1) ? static_cast<float>(gate_packed >> 4) : static_cast<float>(gate_packed & 0x0F); \
                    const float up_q = (group & 1) ? static_cast<float>(up_packed >> 4) : static_cast<float>(up_packed & 0x0F); \
                    const float gate_val = gate_d##N * gate_q - gate_min##N; \
                    const float up_val = up_d##N * up_q - up_min##N; \
                    gate_sum##N##a += gate_val * ba; up_sum##N##a += up_val * ba; \
                    gate_sum##N##b += gate_val * bb; up_sum##N##b += up_val * bb; \
                    gate_sum##N##c += gate_val * bc; up_sum##N##c += up_val * bc; \
                    gate_sum##N##d += gate_val * bd; up_sum##N##d += up_val * bd; \
                } while (0)
                HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(0);
                HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC(1);
#undef HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_ACC
            }
        }

#define HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(S, ID, TOK) \
        do { \
            hrx_reduce_wg4_swiglu<64>(gate_sum0##S, up_sum0##S, gate_sum1##S, up_sum1##S, sumsh); \
            if (tid == 0) { \
                char * dst_base = reinterpret_cast<char *>(dst) + (ID) * c.dst_nb1 + (TOK) * c.dst_nb2; \
                const float silu_gate0 = gate_sum0##S / (1.0f + __expf(-gate_sum0##S)); \
                const float silu_gate1 = gate_sum1##S / (1.0f + __expf(-gate_sum1##S)); \
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0##S * silu_gate0; \
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1##S * silu_gate1; \
            } \
            __syncthreads(); \
        } while (0)
        HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(a, id_a, tok_a);
        if (has_b) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(b, id_b, tok_b);
        }
        if (has_c) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(c, id_c, tok_c);
        }
        if (has_d) {
            HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE(d, id_d, tok_d);
        }
#undef HRX_Q4K_SWIGLU_GROUPED_ROW2_ROUTE4_STORE
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const uint32_t * counts,
        const uint32_t * routes,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_grouped_constants c) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long expert = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 + 1 >= c.rows || expert >= c.n_experts) {
        return;
    }

    const uint32_t count = counts[expert];
    if (count == 0) {
        return;
    }

    __shared__ float sumsh[8 * 4 * (64 / 32)];
    const char * gate_expert_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2;
    const char * up_expert_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2;
    const char * gate_row0_base = gate_expert_base + row0 * c.gate_nb1;
    const char * gate_row1_base = gate_row0_base + c.gate_nb1;
    const char * up_row0_base = up_expert_base + row0 * c.up_nb1;
    const char * up_row1_base = up_row0_base + c.up_nb1;

    const int block_lane = tid & 63;
    const int group = block_lane >> 3;
    const int lane = (block_lane & 7) << 2;
    const long long blocks_per_row = c.k / 256;
    const uint32_t * expert_routes = routes + expert * c.route_capacity;

    for (uint32_t route_idx = 0; route_idx < count; route_idx += 8) {
        const uint32_t route_a = expert_routes[route_idx];
        const uint32_t route_b = route_idx + 1 < count ? expert_routes[route_idx + 1] : route_a;
        const uint32_t route_c = route_idx + 2 < count ? expert_routes[route_idx + 2] : route_a;
        const uint32_t route_d = route_idx + 3 < count ? expert_routes[route_idx + 3] : route_a;
        const uint32_t route_e = route_idx + 4 < count ? expert_routes[route_idx + 4] : route_a;
        const uint32_t route_f = route_idx + 5 < count ? expert_routes[route_idx + 5] : route_a;
        const uint32_t route_g = route_idx + 6 < count ? expert_routes[route_idx + 6] : route_a;
        const uint32_t route_h = route_idx + 7 < count ? expert_routes[route_idx + 7] : route_a;
        const bool has_b = route_idx + 1 < count;
        const bool has_c = route_idx + 2 < count;
        const bool has_d = route_idx + 3 < count;
        const bool has_e = route_idx + 4 < count;
        const bool has_f = route_idx + 5 < count;
        const bool has_g = route_idx + 6 < count;
        const bool has_h = route_idx + 7 < count;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(S) \
        const long long id_##S = static_cast<long long>(route_##S & 7u); \
        const long long tok_##S = static_cast<long long>(route_##S >> 3); \
        const char * src1_##S = reinterpret_cast<const char *>(src1) + id_##S * c.src1_nb1 + tok_##S * c.src1_nb2
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(a);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(b);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(c);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(d);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(e);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(f);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(g);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL(h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_COL

        float gate_sum0a = 0.0f, up_sum0a = 0.0f, gate_sum1a = 0.0f, up_sum1a = 0.0f;
        float gate_sum0b = 0.0f, up_sum0b = 0.0f, gate_sum1b = 0.0f, up_sum1b = 0.0f;
        float gate_sum0c = 0.0f, up_sum0c = 0.0f, gate_sum1c = 0.0f, up_sum1c = 0.0f;
        float gate_sum0d = 0.0f, up_sum0d = 0.0f, gate_sum1d = 0.0f, up_sum1d = 0.0f;
        float gate_sum0e = 0.0f, up_sum0e = 0.0f, gate_sum1e = 0.0f, up_sum1e = 0.0f;
        float gate_sum0f = 0.0f, up_sum0f = 0.0f, gate_sum1f = 0.0f, up_sum1f = 0.0f;
        float gate_sum0g = 0.0f, up_sum0g = 0.0f, gate_sum1g = 0.0f, up_sum1g = 0.0f;
        float gate_sum0h = 0.0f, up_sum0h = 0.0f, gate_sum1h = 0.0f, up_sum1h = 0.0f;

        for (long long block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const hrx_block_q4_K_id_swiglu * gate_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * gate_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                gate_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block0 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row0_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
            const hrx_block_q4_K_id_swiglu * up_block1 = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
                up_row1_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

            uint8_t gate_sc0 = 0, gate_m0 = 0, gate_sc1 = 0, gate_m1 = 0;
            uint8_t up_sc0 = 0, up_m0 = 0, up_sc1 = 0, up_m1 = 0;
            hrx_get_scale_min_k4_id_swiglu(group, gate_block0->scales, &gate_sc0, &gate_m0);
            hrx_get_scale_min_k4_id_swiglu(group, gate_block1->scales, &gate_sc1, &gate_m1);
            hrx_get_scale_min_k4_id_swiglu(group, up_block0->scales, &up_sc0, &up_m0);
            hrx_get_scale_min_k4_id_swiglu(group, up_block1->scales, &up_sc1, &up_m1);

            const float gate_d0 = __half2float(__ushort_as_half(gate_block0->d)) * static_cast<float>(gate_sc0);
            const float gate_d1 = __half2float(__ushort_as_half(gate_block1->d)) * static_cast<float>(gate_sc1);
            const float up_d0 = __half2float(__ushort_as_half(up_block0->d)) * static_cast<float>(up_sc0);
            const float up_d1 = __half2float(__ushort_as_half(up_block1->d)) * static_cast<float>(up_sc1);
            const float gate_min0 = __half2float(__ushort_as_half(gate_block0->dmin)) * static_cast<float>(gate_m0);
            const float gate_min1 = __half2float(__ushort_as_half(gate_block1->dmin)) * static_cast<float>(gate_m1);
            const float up_min0 = __half2float(__ushort_as_half(up_block0->dmin)) * static_cast<float>(up_m0);
            const float up_min1 = __half2float(__ushort_as_half(up_block1->dmin)) * static_cast<float>(up_m1);
            const long long src_base = block_idx * 256 + group * 32 + lane;
            const int qs_base = (group >> 1) * 32 + lane;
            const uint32_t gate_packed_qs0 = *reinterpret_cast<const uint32_t *>(gate_block0->qs + qs_base);
            const uint32_t gate_packed_qs1 = *reinterpret_cast<const uint32_t *>(gate_block1->qs + qs_base);
            const uint32_t up_packed_qs0 = *reinterpret_cast<const uint32_t *>(up_block0->qs + qs_base);
            const uint32_t up_packed_qs1 = *reinterpret_cast<const uint32_t *>(up_block1->qs + qs_base);

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(S, HAS) \
            float4 b4_##S; \
            if (HAS) { \
                b4_##S = *reinterpret_cast<const float4 *>(src1_##S + src_base * sizeof(float)); \
            } else { \
                b4_##S = { 0.0f, 0.0f, 0.0f, 0.0f }; \
            }
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(a, true);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(b, has_b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(c, has_c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(d, has_d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(e, has_e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(f, has_f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(g, has_g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4(h, has_h);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD4

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(N, J) \
                do { \
                    const uint8_t gate_packed = static_cast<uint8_t>(gate_packed_qs##N >> ((J) * 8)); \
                    const uint8_t up_packed = static_cast<uint8_t>(up_packed_qs##N >> ((J) * 8)); \
                    const float gate_q = (group & 1) ? static_cast<float>(gate_packed >> 4) : static_cast<float>(gate_packed & 0x0F); \
                    const float up_q = (group & 1) ? static_cast<float>(up_packed >> 4) : static_cast<float>(up_packed & 0x0F); \
                    const float gate_val = gate_d##N * gate_q - gate_min##N; \
                    const float up_val = up_d##N * up_q - up_min##N; \
                    gate_sum##N##a += gate_val * ba; up_sum##N##a += up_val * ba; \
                    gate_sum##N##b += gate_val * bb; up_sum##N##b += up_val * bb; \
                    gate_sum##N##c += gate_val * bc; up_sum##N##c += up_val * bc; \
                    gate_sum##N##d += gate_val * bd; up_sum##N##d += up_val * bd; \
                    gate_sum##N##e += gate_val * be; up_sum##N##e += up_val * be; \
                    gate_sum##N##f += gate_val * bf; up_sum##N##f += up_val * bf; \
                    gate_sum##N##g += gate_val * bg; up_sum##N##g += up_val * bg; \
                    gate_sum##N##h += gate_val * bh; up_sum##N##h += up_val * bh; \
                } while (0)
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP(J, FIELD) \
            do { \
                const float ba = b4_a.FIELD; \
                const float bb = b4_b.FIELD; \
                const float bc = b4_c.FIELD; \
                const float bd = b4_d.FIELD; \
                const float be = b4_e.FIELD; \
                const float bf = b4_f.FIELD; \
                const float bg = b4_g.FIELD; \
                const float bh = b4_h.FIELD; \
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(0, J); \
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC(1, J); \
            } while (0)
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP(0, x);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP(1, y);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP(2, z);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP(3, w);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_ACC
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_STEP
        }

        const unsigned int lane = tid & (warpSize - 1);
        const unsigned int wave = tid / warpSize;
        constexpr int waves = 64 / 32;
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(S) \
        do { \
            gate_sum0##S += __shfl_down(gate_sum0##S, offset); \
            up_sum0##S += __shfl_down(up_sum0##S, offset); \
            gate_sum1##S += __shfl_down(gate_sum1##S, offset); \
            up_sum1##S += __shfl_down(up_sum1##S, offset); \
        } while (0)
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(a);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(b);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(c);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(d);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(e);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(f);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(g);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(h);
        }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(S, R) \
        do { \
            sumsh[wave + ((R) * 4 + 0) * waves] = gate_sum0##S; \
            sumsh[wave + ((R) * 4 + 1) * waves] = up_sum0##S; \
            sumsh[wave + ((R) * 4 + 2) * waves] = gate_sum1##S; \
            sumsh[wave + ((R) * 4 + 3) * waves] = up_sum1##S; \
        } while (0)
        if (lane == 0) {
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(a, 0);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(b, 1);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(c, 2);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(d, 3);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(e, 4);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(f, 5);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(g, 6);
            HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE(h, 7);
        }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SAVE
        __syncthreads();

#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(S, R) \
        do { \
            gate_sum0##S = lane < waves ? sumsh[lane + ((R) * 4 + 0) * waves] : 0.0f; \
            up_sum0##S = lane < waves ? sumsh[lane + ((R) * 4 + 1) * waves] : 0.0f; \
            gate_sum1##S = lane < waves ? sumsh[lane + ((R) * 4 + 2) * waves] : 0.0f; \
            up_sum1##S = lane < waves ? sumsh[lane + ((R) * 4 + 3) * waves] : 0.0f; \
        } while (0)
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(a, 0);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(b, 1);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(c, 2);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(d, 3);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(e, 4);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(f, 5);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(g, 6);
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD(h, 7);
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_LOAD
        if (wave == 0) {
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(S) \
            do { \
                gate_sum0##S += __shfl_down(gate_sum0##S, offset); \
                up_sum0##S += __shfl_down(up_sum0##S, offset); \
                gate_sum1##S += __shfl_down(gate_sum1##S, offset); \
                up_sum1##S += __shfl_down(up_sum1##S, offset); \
            } while (0)
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(a);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(b);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(c);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(d);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(e);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(f);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(g);
                HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL(h);
            }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_SHFL
        }
#define HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(S, ID, TOK) \
        do { \
            if (tid == 0) { \
                char * dst_base = reinterpret_cast<char *>(dst) + (ID) * c.dst_nb1 + (TOK) * c.dst_nb2; \
                const float silu_gate0 = gate_sum0##S / (1.0f + __expf(-gate_sum0##S)); \
                const float silu_gate1 = gate_sum1##S / (1.0f + __expf(-gate_sum1##S)); \
                *reinterpret_cast<float *>(dst_base + row0 * sizeof(float)) = up_sum0##S * silu_gate0; \
                *reinterpret_cast<float *>(dst_base + (row0 + 1) * sizeof(float)) = up_sum1##S * silu_gate1; \
            } \
        } while (0)
        HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(a, id_a, tok_a);
        if (has_b) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(b, id_b, tok_b); }
        if (has_c) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(c, id_c, tok_c); }
        if (has_d) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(d, id_d, tok_d); }
        if (has_e) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(e, id_e, tok_e); }
        if (has_f) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(f, id_f, tok_f); }
        if (has_g) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(g, id_g, tok_g); }
        if (has_h) { HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE(h, id_h, tok_h); }
#undef HRX_Q4K_SWIGLU_ROW2_ROUTE8_STORE
        __syncthreads();
    }
}

extern "C" __global__ void hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
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

    __shared__ float gate_sumsh[2];
    __shared__ float up_sumsh[2];

    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;

    float gate_sum = 0.0f;
    float up_sum = 0.0f;

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

    (void) blocks_per_row;
    #pragma unroll
    for (int block_iter = 0; block_iter < 2; ++block_iter) {
        const long long block_idx = static_cast<long long>(block_slot + block_iter * 4);
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc0 = 0;
        uint8_t gate_m0 = 0;
        uint8_t gate_sc1 = 0;
        uint8_t gate_m1 = 0;
        uint8_t gate_sc2 = 0;
        uint8_t gate_m2 = 0;
        uint8_t gate_sc3 = 0;
        uint8_t gate_m3 = 0;
        uint8_t up_sc0 = 0;
        uint8_t up_m0 = 0;
        uint8_t up_sc1 = 0;
        uint8_t up_m1 = 0;
        uint8_t up_sc2 = 0;
        uint8_t up_m2 = 0;
        uint8_t up_sc3 = 0;
        uint8_t up_m3 = 0;
        hrx_get_scale_min_k4_id_swiglu(g0, gate_block->scales, &gate_sc0, &gate_m0);
        hrx_get_scale_min_k4_id_swiglu(g1, gate_block->scales, &gate_sc1, &gate_m1);
        hrx_get_scale_min_k4_id_swiglu(g2, gate_block->scales, &gate_sc2, &gate_m2);
        hrx_get_scale_min_k4_id_swiglu(g3, gate_block->scales, &gate_sc3, &gate_m3);
        hrx_get_scale_min_k4_id_swiglu(g0, up_block->scales, &up_sc0, &up_m0);
        hrx_get_scale_min_k4_id_swiglu(g1, up_block->scales, &up_sc1, &up_m1);
        hrx_get_scale_min_k4_id_swiglu(g2, up_block->scales, &up_sc2, &up_m2);
        hrx_get_scale_min_k4_id_swiglu(g3, up_block->scales, &up_sc3, &up_m3);

        const float gate_d = __half2float(__ushort_as_half(gate_block->d));
        const float gate_dmin = __half2float(__ushort_as_half(gate_block->dmin));
        const float gate_d0 = gate_d * static_cast<float>(gate_sc0);
        const float gate_d1 = gate_d * static_cast<float>(gate_sc1);
        const float gate_d2 = gate_d * static_cast<float>(gate_sc2);
        const float gate_d3 = gate_d * static_cast<float>(gate_sc3);
        const float gate_min0 = gate_dmin * static_cast<float>(gate_m0);
        const float gate_min1 = gate_dmin * static_cast<float>(gate_m1);
        const float gate_min2 = gate_dmin * static_cast<float>(gate_m2);
        const float gate_min3 = gate_dmin * static_cast<float>(gate_m3);

        const float up_d = __half2float(__ushort_as_half(up_block->d));
        const float up_dmin = __half2float(__ushort_as_half(up_block->dmin));
        const float up_d0 = up_d * static_cast<float>(up_sc0);
        const float up_d1 = up_d * static_cast<float>(up_sc1);
        const float up_d2 = up_d * static_cast<float>(up_sc2);
        const float up_d3 = up_d * static_cast<float>(up_sc3);
        const float up_min0 = up_dmin * static_cast<float>(up_m0);
        const float up_min1 = up_dmin * static_cast<float>(up_m1);
        const float up_min2 = up_dmin * static_cast<float>(up_m2);
        const float up_min3 = up_dmin * static_cast<float>(up_m3);

        const long long src_base = block_idx * 256 + y_offset;
        const unsigned long long gate_q01 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(gate_block->qs + q_offset));
        const unsigned long long gate_q23 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(gate_block->qs + q_offset + 64));
        const unsigned long long up_q01 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(up_block->qs + q_offset));
        const unsigned long long up_q23 = static_cast<unsigned long long>(
            *reinterpret_cast<const uint32_t *>(up_block->qs + q_offset + 64));
        const float4 y0 = *reinterpret_cast<const float4 *>(src1_col + src_base * sizeof(float));
        const float4 y1 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 32) * sizeof(float));
        const float4 y2 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 128) * sizeof(float));
        const float4 y3 = *reinterpret_cast<const float4 *>(src1_col + (src_base + 160) * sizeof(float));

        gate_sum += (gate_d0 * hrx_q4_k_swiglu_q_from_pack<0>(gate_q01, false) - gate_min0) * y0.x;
        gate_sum += (gate_d0 * hrx_q4_k_swiglu_q_from_pack<1>(gate_q01, false) - gate_min0) * y0.y;
        gate_sum += (gate_d0 * hrx_q4_k_swiglu_q_from_pack<2>(gate_q01, false) - gate_min0) * y0.z;
        gate_sum += (gate_d0 * hrx_q4_k_swiglu_q_from_pack<3>(gate_q01, false) - gate_min0) * y0.w;
        gate_sum += (gate_d1 * hrx_q4_k_swiglu_q_from_pack<0>(gate_q01, true) - gate_min1) * y1.x;
        gate_sum += (gate_d1 * hrx_q4_k_swiglu_q_from_pack<1>(gate_q01, true) - gate_min1) * y1.y;
        gate_sum += (gate_d1 * hrx_q4_k_swiglu_q_from_pack<2>(gate_q01, true) - gate_min1) * y1.z;
        gate_sum += (gate_d1 * hrx_q4_k_swiglu_q_from_pack<3>(gate_q01, true) - gate_min1) * y1.w;
        gate_sum += (gate_d2 * hrx_q4_k_swiglu_q_from_pack<0>(gate_q23, false) - gate_min2) * y2.x;
        gate_sum += (gate_d2 * hrx_q4_k_swiglu_q_from_pack<1>(gate_q23, false) - gate_min2) * y2.y;
        gate_sum += (gate_d2 * hrx_q4_k_swiglu_q_from_pack<2>(gate_q23, false) - gate_min2) * y2.z;
        gate_sum += (gate_d2 * hrx_q4_k_swiglu_q_from_pack<3>(gate_q23, false) - gate_min2) * y2.w;
        gate_sum += (gate_d3 * hrx_q4_k_swiglu_q_from_pack<0>(gate_q23, true) - gate_min3) * y3.x;
        gate_sum += (gate_d3 * hrx_q4_k_swiglu_q_from_pack<1>(gate_q23, true) - gate_min3) * y3.y;
        gate_sum += (gate_d3 * hrx_q4_k_swiglu_q_from_pack<2>(gate_q23, true) - gate_min3) * y3.z;
        gate_sum += (gate_d3 * hrx_q4_k_swiglu_q_from_pack<3>(gate_q23, true) - gate_min3) * y3.w;

        up_sum += (up_d0 * hrx_q4_k_swiglu_q_from_pack<0>(up_q01, false) - up_min0) * y0.x;
        up_sum += (up_d0 * hrx_q4_k_swiglu_q_from_pack<1>(up_q01, false) - up_min0) * y0.y;
        up_sum += (up_d0 * hrx_q4_k_swiglu_q_from_pack<2>(up_q01, false) - up_min0) * y0.z;
        up_sum += (up_d0 * hrx_q4_k_swiglu_q_from_pack<3>(up_q01, false) - up_min0) * y0.w;
        up_sum += (up_d1 * hrx_q4_k_swiglu_q_from_pack<0>(up_q01, true) - up_min1) * y1.x;
        up_sum += (up_d1 * hrx_q4_k_swiglu_q_from_pack<1>(up_q01, true) - up_min1) * y1.y;
        up_sum += (up_d1 * hrx_q4_k_swiglu_q_from_pack<2>(up_q01, true) - up_min1) * y1.z;
        up_sum += (up_d1 * hrx_q4_k_swiglu_q_from_pack<3>(up_q01, true) - up_min1) * y1.w;
        up_sum += (up_d2 * hrx_q4_k_swiglu_q_from_pack<0>(up_q23, false) - up_min2) * y2.x;
        up_sum += (up_d2 * hrx_q4_k_swiglu_q_from_pack<1>(up_q23, false) - up_min2) * y2.y;
        up_sum += (up_d2 * hrx_q4_k_swiglu_q_from_pack<2>(up_q23, false) - up_min2) * y2.z;
        up_sum += (up_d2 * hrx_q4_k_swiglu_q_from_pack<3>(up_q23, false) - up_min2) * y2.w;
        up_sum += (up_d3 * hrx_q4_k_swiglu_q_from_pack<0>(up_q23, true) - up_min3) * y3.x;
        up_sum += (up_d3 * hrx_q4_k_swiglu_q_from_pack<1>(up_q23, true) - up_min3) * y3.y;
        up_sum += (up_d3 * hrx_q4_k_swiglu_q_from_pack<2>(up_q23, true) - up_min3) * y3.z;
        up_sum += (up_d3 * hrx_q4_k_swiglu_q_from_pack<3>(up_q23, true) - up_min3) * y3.w;
    }

    gate_sum = hrx_reduce_wg_swiglu<64>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_wg_swiglu<64>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}

static __global__ void hrx_mul_mat_id_q4_k_swiglu_rows2_x16_wg32_f32(
        const hrx_block_q4_K_id_swiglu * gate,
        const hrx_block_q4_K_id_swiglu * up,
        const float * src1,
        const int * ids,
        float * dst,
        hrx_mul_mat_id_q4_k_swiglu_constants c) {
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
    const char * gate_row_base = reinterpret_cast<const char *>(gate) + expert * c.gate_nb2 + row * c.gate_nb1;
    const char * up_row_base = reinterpret_cast<const char *>(up) + expert * c.up_nb2 + row * c.up_nb1;
    const char * src1_col = reinterpret_cast<const char *>(src1) + id_pos * c.src1_nb1 + token * c.src1_nb2;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;

    #pragma unroll
    for (int iter = 0; iter < 16; ++iter) {
        const long long block_idx = iter >> 1;
        const int group = ((iter & 1) << 2) + static_cast<int>(lane >> 2);
        const int group_offset = static_cast<int>(lane & 3) << 3;
        const long long col = static_cast<long long>(iter) * 128 + static_cast<long long>(lane) * 8;
        const hrx_block_q4_K_id_swiglu * gate_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            gate_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));
        const hrx_block_q4_K_id_swiglu * up_block = reinterpret_cast<const hrx_block_q4_K_id_swiglu *>(
            up_row_base + block_idx * sizeof(hrx_block_q4_K_id_swiglu));

        uint8_t gate_sc = 0;
        uint8_t gate_m = 0;
        uint8_t up_sc = 0;
        uint8_t up_m = 0;
        hrx_get_scale_min_k4_id_swiglu(group, gate_block->scales, &gate_sc, &gate_m);
        hrx_get_scale_min_k4_id_swiglu(group, up_block->scales, &up_sc, &up_m);
        const float gate_d = __half2float(__ushort_as_half(gate_block->d)) * static_cast<float>(gate_sc);
        const float gate_min = __half2float(__ushort_as_half(gate_block->dmin)) * static_cast<float>(gate_m);
        const float up_d = __half2float(__ushort_as_half(up_block->d)) * static_cast<float>(up_sc);
        const float up_min = __half2float(__ushort_as_half(up_block->dmin)) * static_cast<float>(up_m);
        const int qs_base = (group >> 1) * 32 + group_offset;

        const bool high = (group & 1) != 0;
        const unsigned long long gate_qpack = *reinterpret_cast<const unsigned long long *>(gate_block->qs + qs_base);
        const unsigned long long up_qpack = *reinterpret_cast<const unsigned long long *>(up_block->qs + qs_base);
        const float4 b0 = *reinterpret_cast<const float4 *>(src1_col + col * sizeof(float));
        const float4 b1 = *reinterpret_cast<const float4 *>(src1_col + (col + 4) * sizeof(float));

        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<0>(gate_qpack, high) - gate_min) * b0.x;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<1>(gate_qpack, high) - gate_min) * b0.y;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<2>(gate_qpack, high) - gate_min) * b0.z;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<3>(gate_qpack, high) - gate_min) * b0.w;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<4>(gate_qpack, high) - gate_min) * b1.x;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<5>(gate_qpack, high) - gate_min) * b1.y;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<6>(gate_qpack, high) - gate_min) * b1.z;
        gate_sum += (gate_d * hrx_q4_k_swiglu_q_from_pack<7>(gate_qpack, high) - gate_min) * b1.w;

        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<0>(up_qpack, high) - up_min) * b0.x;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<1>(up_qpack, high) - up_min) * b0.y;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<2>(up_qpack, high) - up_min) * b0.z;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<3>(up_qpack, high) - up_min) * b0.w;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<4>(up_qpack, high) - up_min) * b1.x;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<5>(up_qpack, high) - up_min) * b1.y;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<6>(up_qpack, high) - up_min) * b1.z;
        up_sum += (up_d * hrx_q4_k_swiglu_q_from_pack<7>(up_qpack, high) - up_min) * b1.w;
    }

    for (int offset = 8; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset, 16);
        up_sum += __shfl_down(up_sum, offset, 16);
    }

    if (lane == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        *reinterpret_cast<float *>(
            reinterpret_cast<char *>(dst) + row * sizeof(float) + id_pos * c.dst_nb1 + token * c.dst_nb2) =
            up_sum * silu_gate;
    }
}
