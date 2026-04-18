#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float hrx_bf16_swiglu_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define HRX_BF16_SWIGLU_HAS_GFX11_WMMA 1

typedef short hrx_bf16_swiglu_wmma_vec16 __attribute__((ext_vector_type(16)));
typedef float hrx_f32_swiglu_wmma_vec8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ uint16_t hrx_f32_to_bf16_swiglu_bits_rne(float value) {
    union {
        float f;
        uint32_t u;
    } bits = { value };
    const uint32_t lsb = (bits.u >> 16) & 1u;
    bits.u += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits.u >> 16);
}

static __device__ __forceinline__ uint16_t hrx_bf16_swiglu_wmma_swizzle_half(uint32_t packed, int idx) {
    return static_cast<uint16_t>((packed >> (idx * 16)) & 0xffffu);
}

static __device__ __forceinline__ hrx_bf16_swiglu_wmma_vec16 hrx_duplicate_bf16_swiglu_wmma_input(
        uint16_t x0, uint16_t x1, uint16_t x2, uint16_t x3,
        uint16_t x4, uint16_t x5, uint16_t x6, uint16_t x7) {
    constexpr int SWAP16_CTRL = (16 << 10) | 0x1f;
    const uint32_t p0 = static_cast<uint32_t>(x0) | (static_cast<uint32_t>(x1) << 16);
    const uint32_t p1 = static_cast<uint32_t>(x2) | (static_cast<uint32_t>(x3) << 16);
    const uint32_t p2 = static_cast<uint32_t>(x4) | (static_cast<uint32_t>(x5) << 16);
    const uint32_t p3 = static_cast<uint32_t>(x6) | (static_cast<uint32_t>(x7) << 16);
    const uint32_t s0 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p0), SWAP16_CTRL));
    const uint32_t s1 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p1), SWAP16_CTRL));
    const uint32_t s2 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p2), SWAP16_CTRL));
    const uint32_t s3 = static_cast<uint32_t>(__builtin_amdgcn_ds_swizzle(static_cast<int32_t>(p3), SWAP16_CTRL));

    hrx_bf16_swiglu_wmma_vec16 result;
    result[0] = static_cast<short>(x0);
    result[1] = static_cast<short>(x1);
    result[2] = static_cast<short>(x2);
    result[3] = static_cast<short>(x3);
    result[4] = static_cast<short>(x4);
    result[5] = static_cast<short>(x5);
    result[6] = static_cast<short>(x6);
    result[7] = static_cast<short>(x7);
    result[8] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s0, 0));
    result[9] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s0, 1));
    result[10] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s1, 0));
    result[11] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s1, 1));
    result[12] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s2, 0));
    result[13] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s2, 1));
    result[14] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s3, 0));
    result[15] = static_cast<short>(hrx_bf16_swiglu_wmma_swizzle_half(s3, 1));
    return result;
}

static __device__ __forceinline__ hrx_bf16_swiglu_wmma_vec16 hrx_load_bf16_swiglu_wmma_a_row_major(
        const uint16_t * base, int ldm, unsigned int lane) {
    const int row = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const uint16_t * ptr = base + row * ldm + k_base;
    return hrx_duplicate_bf16_swiglu_wmma_input(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

static __device__ __forceinline__ hrx_bf16_swiglu_wmma_vec16 hrx_load_bf16_swiglu_wmma_b_col_major(
        const float * base, int ldm, unsigned int lane) {
    const int col = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const float * ptr = base + col * ldm + k_base;
    return hrx_duplicate_bf16_swiglu_wmma_input(
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[0]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[1]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[2]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[3]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[4]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[5]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[6]),
        hrx_f32_to_bf16_swiglu_bits_rne(ptr[7]));
}

static __device__ __forceinline__ hrx_f32_swiglu_wmma_vec8 hrx_wmma_f32_16x16x16_bf16_swiglu(
        hrx_bf16_swiglu_wmma_vec16 a,
        hrx_bf16_swiglu_wmma_vec16 b,
        hrx_f32_swiglu_wmma_vec8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

static __device__ __forceinline__ void hrx_store_bf16_swiglu_wmma_acc_row_major(
        float * dst,
        int ldm,
        hrx_f32_swiglu_wmma_vec8 gate_acc,
        hrx_f32_swiglu_wmma_vec8 up_acc,
        unsigned int lane) {
    const int row_base = static_cast<int>(lane >> 4);
    const int col = static_cast<int>(lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float gate = gate_acc[i];
        const float silu_gate = gate / (1.0f + __expf(-gate));
        dst[col * ldm + row_base + i * 2] = up_acc[i] * silu_gate;
    }
}

#endif

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_bf16_swiglu(float sum, float * shared) {
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
static __device__ __forceinline__ void hrx_reduce8_bf16_swiglu(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
        float & sum4,
        float & sum5,
        float & sum6,
        float & sum7,
        float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    const int waves = (WG_SIZE + warpSize - 1) / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset);
        sum1 += __shfl_down(sum1, offset);
        sum2 += __shfl_down(sum2, offset);
        sum3 += __shfl_down(sum3, offset);
        sum4 += __shfl_down(sum4, offset);
        sum5 += __shfl_down(sum5, offset);
        sum6 += __shfl_down(sum6, offset);
        sum7 += __shfl_down(sum7, offset);
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
        shared[wave + 0 * waves] = sum0;
        shared[wave + 1 * waves] = sum1;
        shared[wave + 2 * waves] = sum2;
        shared[wave + 3 * waves] = sum3;
        shared[wave + 4 * waves] = sum4;
        shared[wave + 5 * waves] = sum5;
        shared[wave + 6 * waves] = sum6;
        shared[wave + 7 * waves] = sum7;
    }
    __syncthreads();

    sum0 = lane < waves ? shared[lane + 0 * waves] : 0.0f;
    sum1 = lane < waves ? shared[lane + 1 * waves] : 0.0f;
    sum2 = lane < waves ? shared[lane + 2 * waves] : 0.0f;
    sum3 = lane < waves ? shared[lane + 3 * waves] : 0.0f;
    sum4 = lane < waves ? shared[lane + 4 * waves] : 0.0f;
    sum5 = lane < waves ? shared[lane + 5 * waves] : 0.0f;
    sum6 = lane < waves ? shared[lane + 6 * waves] : 0.0f;
    sum7 = lane < waves ? shared[lane + 7 * waves] : 0.0f;
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            sum0 += __shfl_down(sum0, offset);
            sum1 += __shfl_down(sum1, offset);
            sum2 += __shfl_down(sum2, offset);
            sum3 += __shfl_down(sum3, offset);
            sum4 += __shfl_down(sum4, offset);
            sum5 += __shfl_down(sum5, offset);
            sum6 += __shfl_down(sum6, offset);
            sum7 += __shfl_down(sum7, offset);
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_bf16_swiglu_f32_impl(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float gate_sumsh[(WG_SIZE + 31) / 32];
    __shared__ float up_sumsh[(WG_SIZE + 31) / 32];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col = src1 + col * k;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (long long i = tid; i < k; i += WG_SIZE) {
        const float b = src1_col[i];
        gate_sum += hrx_bf16_swiglu_to_f32(gate_row[i]) * b;
        up_sum += hrx_bf16_swiglu_to_f32(up_row[i]) * b;
    }

    gate_sum = hrx_reduce_bf16_swiglu<WG_SIZE>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_bf16_swiglu<WG_SIZE>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[col * rows + row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    hrx_mul_mat_vec_bf16_swiglu_f32_impl<256>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_wmma16x16_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
#if HRX_BF16_SWIGLU_HAS_GFX11_WMMA
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 16;
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 16;
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 31u;
    if (row0 + 15 >= rows || col0 + 15 >= cols) {
        return;
    }

    hrx_f32_swiglu_wmma_vec8 gate_acc = {};
    hrx_f32_swiglu_wmma_vec8 up_acc = {};
    for (long long kb = 0; kb < k; kb += 16) {
        const hrx_bf16_swiglu_wmma_vec16 gate_a =
            hrx_load_bf16_swiglu_wmma_a_row_major(gate + row0 * k + kb, static_cast<int>(k), lane);
        const hrx_bf16_swiglu_wmma_vec16 up_a =
            hrx_load_bf16_swiglu_wmma_a_row_major(up + row0 * k + kb, static_cast<int>(k), lane);
        const hrx_bf16_swiglu_wmma_vec16 b =
            hrx_load_bf16_swiglu_wmma_b_col_major(src1 + col0 * k + kb, static_cast<int>(k), lane);
        gate_acc = hrx_wmma_f32_16x16x16_bf16_swiglu(gate_a, b, gate_acc);
        up_acc = hrx_wmma_f32_16x16x16_bf16_swiglu(up_a, b, up_acc);
    }

    hrx_store_bf16_swiglu_wmma_acc_row_major(
        dst + col0 * rows + row0, static_cast<int>(rows), gate_acc, up_acc, lane);
#else
    (void) gate;
    (void) up;
    (void) src1;
    (void) dst;
    (void) k;
    (void) rows;
    (void) cols;
#endif
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_wg128_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    hrx_mul_mat_vec_bf16_swiglu_f32_impl<128>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_wg64_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    hrx_mul_mat_vec_bf16_swiglu_f32_impl<64>(gate, up, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_cols1_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows) {
        return;
    }
    (void) cols;

    __shared__ float gate_sumsh[8];
    __shared__ float up_sumsh[8];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float b = src1[i];
        gate_sum += hrx_bf16_swiglu_to_f32(gate_row[i]) * b;
        up_sum += hrx_bf16_swiglu_to_f32(up_row[i]) * b;
    }

    gate_sum = hrx_reduce_bf16_swiglu<256>(gate_sum, gate_sumsh);
    up_sum = hrx_reduce_bf16_swiglu<256>(up_sum, up_sumsh);

    if (tid == 0) {
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_rows2_cols1_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    __shared__ float sumsh[8 * ((256 + 31) / 32)];

    const uint16_t * gate_row0 = gate + row0 * k;
    const uint16_t * up_row0 = up + row0 * k;
    const uint16_t * gate_row1 = gate + row1 * k;
    const uint16_t * up_row1 = up + row1 * k;
    const bool have_row1 = row1 < rows;
    float gate0 = 0.0f;
    float up0 = 0.0f;
    float gate1 = 0.0f;
    float up1 = 0.0f;
    float dummy0 = 0.0f;
    float dummy1 = 0.0f;
    float dummy2 = 0.0f;
    float dummy3 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float b = src1[i];
        gate0 += hrx_bf16_swiglu_to_f32(gate_row0[i]) * b;
        up0 += hrx_bf16_swiglu_to_f32(up_row0[i]) * b;
        if (have_row1) {
            gate1 += hrx_bf16_swiglu_to_f32(gate_row1[i]) * b;
            up1 += hrx_bf16_swiglu_to_f32(up_row1[i]) * b;
        }
    }

    hrx_reduce8_bf16_swiglu<256>(
        gate0, up0, gate1, up1,
        dummy0, dummy1, dummy2, dummy3, sumsh);

    if (tid == 0) {
        const float silu_gate0 = gate0 / (1.0f + __expf(-gate0));
        dst[row0] = up0 * silu_gate0;
        if (have_row1) {
            const float silu_gate1 = gate1 / (1.0f + __expf(-gate1));
            dst[row1] = up1 * silu_gate1;
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    (void) cols;
    (void) k;

    __shared__ float rhs[2048];
    __shared__ float gate_partial[8];
    __shared__ float up_partial[8];
    for (unsigned int i = tid; i < 2048; i += 256) {
        rhs[i] = src1[i];
    }
    __syncthreads();

    const unsigned int row_lane = tid >> 6;
    const unsigned int lane = tid & 63;
    const long long row = row0 + static_cast<long long>(row_lane);
    float gate_sum = 0.0f;
    float up_sum = 0.0f;
    if (row < rows) {
        const uint16_t * gate_row = gate + row * 2048;
        const uint16_t * up_row = up + row * 2048;
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const unsigned int i = lane * 2 + static_cast<unsigned int>(iter) * 128;
            const float2 b = *reinterpret_cast<const float2 *>(rhs + i);
            const uint32_t g = *reinterpret_cast<const uint32_t *>(gate_row + i);
            const uint32_t u = *reinterpret_cast<const uint32_t *>(up_row + i);
            gate_sum += hrx_bf16_swiglu_to_f32(static_cast<uint16_t>(g & 0xffffu)) * b.x;
            gate_sum += hrx_bf16_swiglu_to_f32(static_cast<uint16_t>(g >> 16)) * b.y;
            up_sum += hrx_bf16_swiglu_to_f32(static_cast<uint16_t>(u & 0xffffu)) * b.x;
            up_sum += hrx_bf16_swiglu_to_f32(static_cast<uint16_t>(u >> 16)) * b.y;
        }
    }

    const unsigned int sublane = lane & 31;
    const unsigned int subwave = lane >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        gate_sum += __shfl_down(gate_sum, offset, 32);
        up_sum += __shfl_down(up_sum, offset, 32);
    }

    if (sublane == 0) {
        gate_partial[row_lane * 2 + subwave] = gate_sum;
        up_partial[row_lane * 2 + subwave] = up_sum;
    }
    __syncthreads();

    if (lane == 0 && row < rows) {
        gate_sum = gate_partial[row_lane * 2] + gate_partial[row_lane * 2 + 1];
        up_sum = up_partial[row_lane * 2] + up_partial[row_lane * 2 + 1];
        const float silu_gate = gate_sum / (1.0f + __expf(-gate_sum));
        dst[row] = up_sum * silu_gate;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_cols4_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * (256 / 32)];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float up_sum2 = 0.0f;
    float up_sum3 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float g = hrx_bf16_swiglu_to_f32(gate_row[i]);
        const float u = hrx_bf16_swiglu_to_f32(up_row[i]);
        const float b0 = src1_col0[i];
        const float b1 = src1_col0[k + i];
        const float b2 = src1_col0[2 * k + i];
        const float b3 = src1_col0[3 * k + i];
        gate_sum0 += g * b0;
        gate_sum1 += g * b1;
        gate_sum2 += g * b2;
        gate_sum3 += g * b3;
        up_sum0 += u * b0;
        up_sum1 += u * b1;
        up_sum2 += u * b2;
        up_sum3 += u * b3;
    }

    hrx_reduce8_bf16_swiglu<256>(
        gate_sum0, gate_sum1, gate_sum2, gate_sum3,
        up_sum0, up_sum1, up_sum2, up_sum3, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        dst_col0[0] = up_sum0 * silu_gate0;
        dst_col0[rows] = up_sum1 * silu_gate1;
        dst_col0[2 * rows] = up_sum2 * silu_gate2;
        dst_col0[3 * rows] = up_sum3 * silu_gate3;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_cols8_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float gate_sumsh[8 * (256 / 32)];
    __shared__ float up_sumsh[8 * (256 / 32)];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float gate_sum4 = 0.0f;
    float gate_sum5 = 0.0f;
    float gate_sum6 = 0.0f;
    float gate_sum7 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float up_sum2 = 0.0f;
    float up_sum3 = 0.0f;
    float up_sum4 = 0.0f;
    float up_sum5 = 0.0f;
    float up_sum6 = 0.0f;
    float up_sum7 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float g = hrx_bf16_swiglu_to_f32(gate_row[i]);
        const float u = hrx_bf16_swiglu_to_f32(up_row[i]);
        const float b0 = src1_col0[i];
        const float b1 = src1_col0[k + i];
        const float b2 = src1_col0[2 * k + i];
        const float b3 = src1_col0[3 * k + i];
        const float b4 = src1_col0[4 * k + i];
        const float b5 = src1_col0[5 * k + i];
        const float b6 = src1_col0[6 * k + i];
        const float b7 = src1_col0[7 * k + i];
        gate_sum0 += g * b0;
        gate_sum1 += g * b1;
        gate_sum2 += g * b2;
        gate_sum3 += g * b3;
        gate_sum4 += g * b4;
        gate_sum5 += g * b5;
        gate_sum6 += g * b6;
        gate_sum7 += g * b7;
        up_sum0 += u * b0;
        up_sum1 += u * b1;
        up_sum2 += u * b2;
        up_sum3 += u * b3;
        up_sum4 += u * b4;
        up_sum5 += u * b5;
        up_sum6 += u * b6;
        up_sum7 += u * b7;
    }

    hrx_reduce8_bf16_swiglu<256>(
        gate_sum0, gate_sum1, gate_sum2, gate_sum3,
        gate_sum4, gate_sum5, gate_sum6, gate_sum7, gate_sumsh);
    hrx_reduce8_bf16_swiglu<256>(
        up_sum0, up_sum1, up_sum2, up_sum3,
        up_sum4, up_sum5, up_sum6, up_sum7, up_sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        const float silu_gate4 = gate_sum4 / (1.0f + __expf(-gate_sum4));
        const float silu_gate5 = gate_sum5 / (1.0f + __expf(-gate_sum5));
        const float silu_gate6 = gate_sum6 / (1.0f + __expf(-gate_sum6));
        const float silu_gate7 = gate_sum7 / (1.0f + __expf(-gate_sum7));
        dst_col0[0] = up_sum0 * silu_gate0;
        dst_col0[rows] = up_sum1 * silu_gate1;
        dst_col0[2 * rows] = up_sum2 * silu_gate2;
        dst_col0[3 * rows] = up_sum3 * silu_gate3;
        dst_col0[4 * rows] = up_sum4 * silu_gate4;
        dst_col0[5 * rows] = up_sum5 * silu_gate5;
        dst_col0[6 * rows] = up_sum6 * silu_gate6;
        dst_col0[7 * rows] = up_sum7 * silu_gate7;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_rows2_cols8_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row0 = __builtin_amdgcn_workgroup_id_x() * 2;
    const long long row1 = row0 + 1;
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row1 >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float gate_sumsh0[8 * ((32 + 31) / 32)];
    __shared__ float gate_sumsh1[8 * ((32 + 31) / 32)];
    __shared__ float up_sumsh0[8 * ((32 + 31) / 32)];
    __shared__ float up_sumsh1[8 * ((32 + 31) / 32)];

    const uint16_t * gate_row0 = gate + row0 * k;
    const uint16_t * gate_row1 = gate + row1 * k;
    const uint16_t * up_row0 = up + row0 * k;
    const uint16_t * up_row1 = up + row1 * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate0[8] = {};
    float gate1[8] = {};
    float up0[8] = {};
    float up1[8] = {};
    for (long long i = tid; i < k; i += 32) {
        const float g0 = hrx_bf16_swiglu_to_f32(gate_row0[i]);
        const float g1 = hrx_bf16_swiglu_to_f32(gate_row1[i]);
        const float u0 = hrx_bf16_swiglu_to_f32(up_row0[i]);
        const float u1 = hrx_bf16_swiglu_to_f32(up_row1[i]);
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            const float b = src1_col0[static_cast<long long>(c) * k + i];
            gate0[c] += g0 * b;
            gate1[c] += g1 * b;
            up0[c] += u0 * b;
            up1[c] += u1 * b;
        }
    }

    hrx_reduce8_bf16_swiglu<32>(
        gate0[0], gate0[1], gate0[2], gate0[3],
        gate0[4], gate0[5], gate0[6], gate0[7], gate_sumsh0);
    hrx_reduce8_bf16_swiglu<32>(
        gate1[0], gate1[1], gate1[2], gate1[3],
        gate1[4], gate1[5], gate1[6], gate1[7], gate_sumsh1);
    hrx_reduce8_bf16_swiglu<32>(
        up0[0], up0[1], up0[2], up0[3],
        up0[4], up0[5], up0[6], up0[7], up_sumsh0);
    hrx_reduce8_bf16_swiglu<32>(
        up1[0], up1[1], up1[2], up1[3],
        up1[4], up1[5], up1[6], up1[7], up_sumsh1);

    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            float * dst_col = dst + (col0 + c) * rows;
            const float silu_gate0 = gate0[c] / (1.0f + __expf(-gate0[c]));
            const float silu_gate1 = gate1[c] / (1.0f + __expf(-gate1[c]));
            dst_col[row0] = up0[c] * silu_gate0;
            dst_col[row1] = up1[c] * silu_gate1;
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_swiglu_cols16_f32(
        const uint16_t * gate,
        const uint16_t * up,
        const float * src1,
        float * dst,
        long long k,
        long long rows,
        long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 16;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 15 >= cols) {
        return;
    }

    __shared__ float gate_sumsh0[8 * (256 / 32)];
    __shared__ float gate_sumsh1[8 * (256 / 32)];
    __shared__ float up_sumsh0[8 * (256 / 32)];
    __shared__ float up_sumsh1[8 * (256 / 32)];

    const uint16_t * gate_row = gate + row * k;
    const uint16_t * up_row = up + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float gate_sum0 = 0.0f;
    float gate_sum1 = 0.0f;
    float gate_sum2 = 0.0f;
    float gate_sum3 = 0.0f;
    float gate_sum4 = 0.0f;
    float gate_sum5 = 0.0f;
    float gate_sum6 = 0.0f;
    float gate_sum7 = 0.0f;
    float gate_sum8 = 0.0f;
    float gate_sum9 = 0.0f;
    float gate_sum10 = 0.0f;
    float gate_sum11 = 0.0f;
    float gate_sum12 = 0.0f;
    float gate_sum13 = 0.0f;
    float gate_sum14 = 0.0f;
    float gate_sum15 = 0.0f;
    float up_sum0 = 0.0f;
    float up_sum1 = 0.0f;
    float up_sum2 = 0.0f;
    float up_sum3 = 0.0f;
    float up_sum4 = 0.0f;
    float up_sum5 = 0.0f;
    float up_sum6 = 0.0f;
    float up_sum7 = 0.0f;
    float up_sum8 = 0.0f;
    float up_sum9 = 0.0f;
    float up_sum10 = 0.0f;
    float up_sum11 = 0.0f;
    float up_sum12 = 0.0f;
    float up_sum13 = 0.0f;
    float up_sum14 = 0.0f;
    float up_sum15 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float g = hrx_bf16_swiglu_to_f32(gate_row[i]);
        const float u = hrx_bf16_swiglu_to_f32(up_row[i]);
        const float b0 = src1_col0[i];
        const float b1 = src1_col0[k + i];
        const float b2 = src1_col0[2 * k + i];
        const float b3 = src1_col0[3 * k + i];
        const float b4 = src1_col0[4 * k + i];
        const float b5 = src1_col0[5 * k + i];
        const float b6 = src1_col0[6 * k + i];
        const float b7 = src1_col0[7 * k + i];
        const float b8 = src1_col0[8 * k + i];
        const float b9 = src1_col0[9 * k + i];
        const float b10 = src1_col0[10 * k + i];
        const float b11 = src1_col0[11 * k + i];
        const float b12 = src1_col0[12 * k + i];
        const float b13 = src1_col0[13 * k + i];
        const float b14 = src1_col0[14 * k + i];
        const float b15 = src1_col0[15 * k + i];
        gate_sum0 += g * b0;
        gate_sum1 += g * b1;
        gate_sum2 += g * b2;
        gate_sum3 += g * b3;
        gate_sum4 += g * b4;
        gate_sum5 += g * b5;
        gate_sum6 += g * b6;
        gate_sum7 += g * b7;
        gate_sum8 += g * b8;
        gate_sum9 += g * b9;
        gate_sum10 += g * b10;
        gate_sum11 += g * b11;
        gate_sum12 += g * b12;
        gate_sum13 += g * b13;
        gate_sum14 += g * b14;
        gate_sum15 += g * b15;
        up_sum0 += u * b0;
        up_sum1 += u * b1;
        up_sum2 += u * b2;
        up_sum3 += u * b3;
        up_sum4 += u * b4;
        up_sum5 += u * b5;
        up_sum6 += u * b6;
        up_sum7 += u * b7;
        up_sum8 += u * b8;
        up_sum9 += u * b9;
        up_sum10 += u * b10;
        up_sum11 += u * b11;
        up_sum12 += u * b12;
        up_sum13 += u * b13;
        up_sum14 += u * b14;
        up_sum15 += u * b15;
    }

    hrx_reduce8_bf16_swiglu<256>(
        gate_sum0, gate_sum1, gate_sum2, gate_sum3,
        gate_sum4, gate_sum5, gate_sum6, gate_sum7, gate_sumsh0);
    hrx_reduce8_bf16_swiglu<256>(
        gate_sum8, gate_sum9, gate_sum10, gate_sum11,
        gate_sum12, gate_sum13, gate_sum14, gate_sum15, gate_sumsh1);
    hrx_reduce8_bf16_swiglu<256>(
        up_sum0, up_sum1, up_sum2, up_sum3,
        up_sum4, up_sum5, up_sum6, up_sum7, up_sumsh0);
    hrx_reduce8_bf16_swiglu<256>(
        up_sum8, up_sum9, up_sum10, up_sum11,
        up_sum12, up_sum13, up_sum14, up_sum15, up_sumsh1);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        const float silu_gate0 = gate_sum0 / (1.0f + __expf(-gate_sum0));
        const float silu_gate1 = gate_sum1 / (1.0f + __expf(-gate_sum1));
        const float silu_gate2 = gate_sum2 / (1.0f + __expf(-gate_sum2));
        const float silu_gate3 = gate_sum3 / (1.0f + __expf(-gate_sum3));
        const float silu_gate4 = gate_sum4 / (1.0f + __expf(-gate_sum4));
        const float silu_gate5 = gate_sum5 / (1.0f + __expf(-gate_sum5));
        const float silu_gate6 = gate_sum6 / (1.0f + __expf(-gate_sum6));
        const float silu_gate7 = gate_sum7 / (1.0f + __expf(-gate_sum7));
        const float silu_gate8 = gate_sum8 / (1.0f + __expf(-gate_sum8));
        const float silu_gate9 = gate_sum9 / (1.0f + __expf(-gate_sum9));
        const float silu_gate10 = gate_sum10 / (1.0f + __expf(-gate_sum10));
        const float silu_gate11 = gate_sum11 / (1.0f + __expf(-gate_sum11));
        const float silu_gate12 = gate_sum12 / (1.0f + __expf(-gate_sum12));
        const float silu_gate13 = gate_sum13 / (1.0f + __expf(-gate_sum13));
        const float silu_gate14 = gate_sum14 / (1.0f + __expf(-gate_sum14));
        const float silu_gate15 = gate_sum15 / (1.0f + __expf(-gate_sum15));
        dst_col0[0] = up_sum0 * silu_gate0;
        dst_col0[rows] = up_sum1 * silu_gate1;
        dst_col0[2 * rows] = up_sum2 * silu_gate2;
        dst_col0[3 * rows] = up_sum3 * silu_gate3;
        dst_col0[4 * rows] = up_sum4 * silu_gate4;
        dst_col0[5 * rows] = up_sum5 * silu_gate5;
        dst_col0[6 * rows] = up_sum6 * silu_gate6;
        dst_col0[7 * rows] = up_sum7 * silu_gate7;
        dst_col0[8 * rows] = up_sum8 * silu_gate8;
        dst_col0[9 * rows] = up_sum9 * silu_gate9;
        dst_col0[10 * rows] = up_sum10 * silu_gate10;
        dst_col0[11 * rows] = up_sum11 * silu_gate11;
        dst_col0[12 * rows] = up_sum12 * silu_gate12;
        dst_col0[13 * rows] = up_sum13 * silu_gate13;
        dst_col0[14 * rows] = up_sum14 * silu_gate14;
        dst_col0[15 * rows] = up_sum15 * silu_gate15;
    }
}
