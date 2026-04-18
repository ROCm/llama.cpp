#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

static __device__ __forceinline__ float hrx_bf16_to_f32(uint16_t value) {
    union {
        uint32_t u;
        float f;
    } bits = { static_cast<uint32_t>(value) << 16 };
    return bits.f;
}

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1150__) || defined(__gfx1151__)
#define HRX_BF16_HAS_GFX11_WMMA 1

typedef short hrx_bf16_wmma_vec16 __attribute__((ext_vector_type(16)));
typedef float hrx_f32_wmma_vec8 __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ uint16_t hrx_f32_to_bf16_bits_rne(float value) {
    union {
        float f;
        uint32_t u;
    } bits = { value };
    const uint32_t lsb = (bits.u >> 16) & 1u;
    bits.u += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits.u >> 16);
}

static __device__ __forceinline__ uint16_t hrx_bf16_wmma_swizzle_half(uint32_t packed, int idx) {
    return static_cast<uint16_t>((packed >> (idx * 16)) & 0xffffu);
}

static __device__ __forceinline__ hrx_bf16_wmma_vec16 hrx_duplicate_bf16_wmma_input(
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

    hrx_bf16_wmma_vec16 result;
    result[0] = static_cast<short>(x0);
    result[1] = static_cast<short>(x1);
    result[2] = static_cast<short>(x2);
    result[3] = static_cast<short>(x3);
    result[4] = static_cast<short>(x4);
    result[5] = static_cast<short>(x5);
    result[6] = static_cast<short>(x6);
    result[7] = static_cast<short>(x7);
    result[8] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s0, 0));
    result[9] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s0, 1));
    result[10] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s1, 0));
    result[11] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s1, 1));
    result[12] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s2, 0));
    result[13] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s2, 1));
    result[14] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s3, 0));
    result[15] = static_cast<short>(hrx_bf16_wmma_swizzle_half(s3, 1));
    return result;
}

static __device__ __forceinline__ hrx_bf16_wmma_vec16 hrx_load_bf16_wmma_a_row_major(
        const uint16_t * base, int ldm, unsigned int lane) {
    const int row = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const uint16_t * ptr = base + row * ldm + k_base;
    return hrx_duplicate_bf16_wmma_input(
        ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7]);
}

static __device__ __forceinline__ hrx_bf16_wmma_vec16 hrx_load_bf16_wmma_b_col_major(
        const float * base, int ldm, unsigned int lane) {
    const int col = static_cast<int>(lane & 15);
    const int k_base = static_cast<int>(lane >> 4) * 8;
    const float * ptr = base + col * ldm + k_base;
    return hrx_duplicate_bf16_wmma_input(
        hrx_f32_to_bf16_bits_rne(ptr[0]),
        hrx_f32_to_bf16_bits_rne(ptr[1]),
        hrx_f32_to_bf16_bits_rne(ptr[2]),
        hrx_f32_to_bf16_bits_rne(ptr[3]),
        hrx_f32_to_bf16_bits_rne(ptr[4]),
        hrx_f32_to_bf16_bits_rne(ptr[5]),
        hrx_f32_to_bf16_bits_rne(ptr[6]),
        hrx_f32_to_bf16_bits_rne(ptr[7]));
}

static __device__ __forceinline__ hrx_f32_wmma_vec8 hrx_wmma_f32_16x16x16_bf16(
        hrx_bf16_wmma_vec16 a,
        hrx_bf16_wmma_vec16 b,
        hrx_f32_wmma_vec8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

static __device__ __forceinline__ void hrx_store_bf16_wmma_acc_row_major(
        float * dst,
        int ldm,
        hrx_f32_wmma_vec8 acc,
        unsigned int lane) {
    const int row_base = static_cast<int>(lane >> 4);
    const int col = static_cast<int>(lane & 15);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        dst[col * ldm + row_base + i * 2] = acc[i];
    }
}

#endif

template <int WG_SIZE>
static __device__ __forceinline__ float hrx_reduce_bf16(float sum, float * shared) {
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
static __device__ __forceinline__ void hrx_reduce4_bf16(
        float & sum0,
        float & sum1,
        float & sum2,
        float & sum3,
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
static __device__ __forceinline__ void hrx_reduce8_bf16(
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
static __device__ __forceinline__ void hrx_reduce16_bf16(float sum[16], float * shared) {
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    const unsigned int lane = tid & (warpSize - 1);
    const unsigned int wave = tid / warpSize;
    const int waves = (WG_SIZE + warpSize - 1) / warpSize;

    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            sum[i] += __shfl_down(sum[i], offset);
        }
    }
    if (WG_SIZE <= warpSize) {
        return;
    }
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            shared[wave + i * waves] = sum[i];
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        sum[i] = lane < waves ? shared[lane + i * waves] : 0.0f;
    }
    if (wave == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                sum[i] += __shfl_down(sum[i], offset);
            }
        }
    }
}

template <int WG_SIZE>
static __device__ __forceinline__ void hrx_mul_mat_vec_bf16_f32_impl(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col = __builtin_amdgcn_workgroup_id_y();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col >= cols) {
        return;
    }

    __shared__ float sumsh[(WG_SIZE + 31) / 32];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col = src1 + col * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += WG_SIZE) {
        sum += hrx_bf16_to_f32(src0_row[i]) * src1_col[i];
    }

    sum = hrx_reduce_bf16<WG_SIZE>(sum, sumsh);

    if (tid == 0) {
        dst[col * rows + row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_wmma16x16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
#if HRX_BF16_HAS_GFX11_WMMA
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 16;
    const long long col0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 16;
    const unsigned int lane = __builtin_amdgcn_workitem_id_x() & 31u;
    if (row0 + 15 >= rows || col0 + 15 >= cols) {
        return;
    }

    hrx_f32_wmma_vec8 acc = {};
    for (long long kb = 0; kb < k; kb += 16) {
        const hrx_bf16_wmma_vec16 a =
            hrx_load_bf16_wmma_a_row_major(src0 + row0 * k + kb, static_cast<int>(k), lane);
        const hrx_bf16_wmma_vec16 b =
            hrx_load_bf16_wmma_b_col_major(src1 + col0 * k + kb, static_cast<int>(k), lane);
        acc = hrx_wmma_f32_16x16x16_bf16(a, b, acc);
    }

    hrx_store_bf16_wmma_acc_row_major(dst + col0 * rows + row0, static_cast<int>(rows), acc, lane);
#else
    (void) src0;
    (void) src1;
    (void) dst;
    (void) k;
    (void) rows;
    (void) cols;
#endif
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_bf16_f32_impl<256>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_wg128_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_bf16_f32_impl<128>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_wg64_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    hrx_mul_mat_vec_bf16_f32_impl<64>(src0, src1, dst, k, rows, cols);
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_cols1_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows) {
        return;
    }
    (void) cols;

    __shared__ float sumsh[4 * (256 / 32)];

    const uint16_t * src0_row = src0 + row * k;
    float sum = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        sum += hrx_bf16_to_f32(src0_row[i]) * src1[i];
    }

    sum = hrx_reduce_bf16<256>(sum, sumsh);

    if (tid == 0) {
        dst[row] = sum;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows2_cols1_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    __shared__ float sumsh[4 * ((256 + 31) / 32)];

    const uint16_t * src0_row0 = src0 + row0 * k;
    const uint16_t * src0_row1 = src0 + row1 * k;
    const bool have_row1 = row1 < rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float dummy0 = 0.0f;
    float dummy1 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float b = src1[i];
        sum0 += hrx_bf16_to_f32(src0_row0[i]) * b;
        if (have_row1) {
            sum1 += hrx_bf16_to_f32(src0_row1[i]) * b;
        }
    }

    hrx_reduce4_bf16<256>(sum0, sum1, dummy0, dummy1, sumsh);

    if (tid == 0) {
        dst[row0] = sum0;
        if (have_row1) {
            dst[row1] = sum1;
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows2_cols1_wg32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 2;
    const long long row1 = row0 + 1;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row0 >= rows) {
        return;
    }
    (void) cols;

    const uint16_t * src0_row0 = src0 + row0 * k;
    const uint16_t * src0_row1 = src0 + row1 * k;
    const bool have_row1 = row1 < rows;
    float sum0 = 0.0f;
    float sum1 = 0.0f;

    for (long long i = static_cast<long long>(tid) * 2; i < k; i += 64) {
        const float2 b = *reinterpret_cast<const float2 *>(src1 + i);
        const uint32_t a0 = *reinterpret_cast<const uint32_t *>(src0_row0 + i);
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0 & 0xffffu)) * b.x;
        sum0 += hrx_bf16_to_f32(static_cast<uint16_t>(a0 >> 16)) * b.y;
        if (have_row1) {
            const uint32_t a1 = *reinterpret_cast<const uint32_t *>(src0_row1 + i);
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1 & 0xffffu)) * b.x;
            sum1 += hrx_bf16_to_f32(static_cast<uint16_t>(a1 >> 16)) * b.y;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down(sum0, offset, 32);
        sum1 += __shfl_down(sum1, offset, 32);
    }

    if (tid == 0) {
        dst[row0] = sum0;
        if (have_row1) {
            dst[row1] = sum1;
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    (void) cols;
    (void) k;

    __shared__ float rhs[512];
    __shared__ float partial[8];
    for (unsigned int i = tid; i < 512; i += 256) {
        rhs[i] = src1[i];
    }
    __syncthreads();

    const unsigned int row_lane = tid >> 6;
    const unsigned int lane = tid & 63;
    const long long row = row0 + static_cast<long long>(row_lane);
    float sum = 0.0f;
    if (row < rows) {
        const uint16_t * src0_row = src0 + row * 512;
        #pragma unroll
        for (int iter = 0; iter < 4; ++iter) {
            const unsigned int i = lane * 2 + static_cast<unsigned int>(iter) * 128;
            const uint32_t a = *reinterpret_cast<const uint32_t *>(src0_row + i);
            const float2 b = *reinterpret_cast<const float2 *>(rhs + i);
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a & 0xffffu)) * b.x;
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a >> 16)) * b.y;
        }
    }

    const unsigned int sublane = lane & 31;
    const unsigned int subwave = lane >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (sublane == 0) {
        partial[row_lane * 2 + subwave] = sum;
    }
    __syncthreads();

    if (lane == 0 && row < rows) {
        dst[row] = partial[row_lane * 2] + partial[row_lane * 2 + 1];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    (void) cols;
    (void) k;

    __shared__ float rhs[2048];
    __shared__ float partial[8];
    for (unsigned int i = tid; i < 2048; i += 256) {
        rhs[i] = src1[i];
    }
    __syncthreads();

    const unsigned int row_lane = tid >> 6;
    const unsigned int lane = tid & 63;
    const long long row = row0 + static_cast<long long>(row_lane);
    float sum = 0.0f;
    if (row < rows) {
        const uint16_t * src0_row = src0 + row * 2048;
#pragma unroll
        for (int iter = 0; iter < 16; ++iter) {
            const unsigned int i = lane * 2 + static_cast<unsigned int>(iter) * 128;
            const uint32_t a = *reinterpret_cast<const uint32_t *>(src0_row + i);
            const float2 b = *reinterpret_cast<const float2 *>(rhs + i);
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a & 0xffffu)) * b.x;
            sum += hrx_bf16_to_f32(static_cast<uint16_t>(a >> 16)) * b.y;
        }
    }

    const unsigned int sublane = lane & 31;
    const unsigned int subwave = lane >> 5;
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down(sum, offset, 32);
    }

    if (sublane == 0) {
        partial[row_lane * 2 + subwave] = sum;
    }
    __syncthreads();

    if (lane == 0 && row < rows) {
        dst[row] = partial[row_lane * 2] + partial[row_lane * 2 + 1];
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_cols4_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 4;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 >= cols) {
        return;
    }

    __shared__ float sumsh[8];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = hrx_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
    }

    hrx_reduce4_bf16<256>(sum0, sum1, sum2, sum3, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_cols8_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 8;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 7 >= cols) {
        return;
    }

    __shared__ float sumsh[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = hrx_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
    }

    hrx_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
        dst_col0[4 * rows] = sum4;
        dst_col0[5 * rows] = sum5;
        dst_col0[6 * rows] = sum6;
        dst_col0[7 * rows] = sum7;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_cols16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 16;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 15 >= cols) {
        return;
    }

    __shared__ float sumsh0[8 * ((256 + 31) / 32)];
    __shared__ float sumsh1[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    float sum8 = 0.0f;
    float sum9 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum14 = 0.0f;
    float sum15 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = hrx_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
        sum8 += a * src1_col0[8 * k + i];
        sum9 += a * src1_col0[9 * k + i];
        sum10 += a * src1_col0[10 * k + i];
        sum11 += a * src1_col0[11 * k + i];
        sum12 += a * src1_col0[12 * k + i];
        sum13 += a * src1_col0[13 * k + i];
        sum14 += a * src1_col0[14 * k + i];
        sum15 += a * src1_col0[15 * k + i];
    }

    hrx_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
    hrx_reduce8_bf16<256>(sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh1);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
        dst_col0[4 * rows] = sum4;
        dst_col0[5 * rows] = sum5;
        dst_col0[6 * rows] = sum6;
        dst_col0[7 * rows] = sum7;
        dst_col0[8 * rows] = sum8;
        dst_col0[9 * rows] = sum9;
        dst_col0[10 * rows] = sum10;
        dst_col0[11 * rows] = sum11;
        dst_col0[12 * rows] = sum12;
        dst_col0[13 * rows] = sum13;
        dst_col0[14 * rows] = sum14;
        dst_col0[15 * rows] = sum15;
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_rows2_cols16_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row0 = __builtin_amdgcn_workgroup_id_x() * 2;
    const long long row1 = row0 + 1;
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 16;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row1 >= rows || col0 + 15 >= cols) {
        return;
    }

    __shared__ float sumsh0[16 * ((32 + 31) / 32)];
    __shared__ float sumsh1[16 * ((32 + 31) / 32)];

    const uint16_t * src0_row0 = src0 + row0 * k;
    const uint16_t * src0_row1 = src0 + row1 * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0[16] = {};
    float sum1[16] = {};
    for (long long i = tid; i < k; i += 32) {
        const float a0 = hrx_bf16_to_f32(src0_row0[i]);
        const float a1 = hrx_bf16_to_f32(src0_row1[i]);
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            const float b = src1_col0[static_cast<long long>(c) * k + i];
            sum0[c] += a0 * b;
            sum1[c] += a1 * b;
        }
    }

    hrx_reduce16_bf16<32>(sum0, sumsh0);
    hrx_reduce16_bf16<32>(sum1, sumsh1);

    if (tid == 0) {
#pragma unroll
        for (int c = 0; c < 16; ++c) {
            float * dst_col = dst + (col0 + c) * rows;
            dst_col[row0] = sum0[c];
            dst_col[row1] = sum1[c];
        }
    }
}

extern "C" __global__ void hrx_mul_mat_vec_bf16_cols32_f32(
        const uint16_t * src0, const float * src1, float * dst,
        long long k, long long rows, long long cols) {
    const long long row = __builtin_amdgcn_workgroup_id_x();
    const long long col0 = __builtin_amdgcn_workgroup_id_y() * 32;
    const unsigned int tid = __builtin_amdgcn_workitem_id_x();
    if (row >= rows || col0 + 31 >= cols) {
        return;
    }

    __shared__ float sumsh0[8 * ((256 + 31) / 32)];
    __shared__ float sumsh1[8 * ((256 + 31) / 32)];
    __shared__ float sumsh2[8 * ((256 + 31) / 32)];
    __shared__ float sumsh3[8 * ((256 + 31) / 32)];

    const uint16_t * src0_row = src0 + row * k;
    const float * src1_col0 = src1 + col0 * k;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    float sum4 = 0.0f;
    float sum5 = 0.0f;
    float sum6 = 0.0f;
    float sum7 = 0.0f;
    float sum8 = 0.0f;
    float sum9 = 0.0f;
    float sum10 = 0.0f;
    float sum11 = 0.0f;
    float sum12 = 0.0f;
    float sum13 = 0.0f;
    float sum14 = 0.0f;
    float sum15 = 0.0f;
    float sum16 = 0.0f;
    float sum17 = 0.0f;
    float sum18 = 0.0f;
    float sum19 = 0.0f;
    float sum20 = 0.0f;
    float sum21 = 0.0f;
    float sum22 = 0.0f;
    float sum23 = 0.0f;
    float sum24 = 0.0f;
    float sum25 = 0.0f;
    float sum26 = 0.0f;
    float sum27 = 0.0f;
    float sum28 = 0.0f;
    float sum29 = 0.0f;
    float sum30 = 0.0f;
    float sum31 = 0.0f;
    for (long long i = tid; i < k; i += 256) {
        const float a = hrx_bf16_to_f32(src0_row[i]);
        sum0 += a * src1_col0[i];
        sum1 += a * src1_col0[k + i];
        sum2 += a * src1_col0[2 * k + i];
        sum3 += a * src1_col0[3 * k + i];
        sum4 += a * src1_col0[4 * k + i];
        sum5 += a * src1_col0[5 * k + i];
        sum6 += a * src1_col0[6 * k + i];
        sum7 += a * src1_col0[7 * k + i];
        sum8 += a * src1_col0[8 * k + i];
        sum9 += a * src1_col0[9 * k + i];
        sum10 += a * src1_col0[10 * k + i];
        sum11 += a * src1_col0[11 * k + i];
        sum12 += a * src1_col0[12 * k + i];
        sum13 += a * src1_col0[13 * k + i];
        sum14 += a * src1_col0[14 * k + i];
        sum15 += a * src1_col0[15 * k + i];
        sum16 += a * src1_col0[16 * k + i];
        sum17 += a * src1_col0[17 * k + i];
        sum18 += a * src1_col0[18 * k + i];
        sum19 += a * src1_col0[19 * k + i];
        sum20 += a * src1_col0[20 * k + i];
        sum21 += a * src1_col0[21 * k + i];
        sum22 += a * src1_col0[22 * k + i];
        sum23 += a * src1_col0[23 * k + i];
        sum24 += a * src1_col0[24 * k + i];
        sum25 += a * src1_col0[25 * k + i];
        sum26 += a * src1_col0[26 * k + i];
        sum27 += a * src1_col0[27 * k + i];
        sum28 += a * src1_col0[28 * k + i];
        sum29 += a * src1_col0[29 * k + i];
        sum30 += a * src1_col0[30 * k + i];
        sum31 += a * src1_col0[31 * k + i];
    }

    hrx_reduce8_bf16<256>(sum0, sum1, sum2, sum3, sum4, sum5, sum6, sum7, sumsh0);
    hrx_reduce8_bf16<256>(sum8, sum9, sum10, sum11, sum12, sum13, sum14, sum15, sumsh1);
    hrx_reduce8_bf16<256>(sum16, sum17, sum18, sum19, sum20, sum21, sum22, sum23, sumsh2);
    hrx_reduce8_bf16<256>(sum24, sum25, sum26, sum27, sum28, sum29, sum30, sum31, sumsh3);

    if (tid == 0) {
        float * dst_col0 = dst + col0 * rows + row;
        dst_col0[0] = sum0;
        dst_col0[rows] = sum1;
        dst_col0[2 * rows] = sum2;
        dst_col0[3 * rows] = sum3;
        dst_col0[4 * rows] = sum4;
        dst_col0[5 * rows] = sum5;
        dst_col0[6 * rows] = sum6;
        dst_col0[7 * rows] = sum7;
        dst_col0[8 * rows] = sum8;
        dst_col0[9 * rows] = sum9;
        dst_col0[10 * rows] = sum10;
        dst_col0[11 * rows] = sum11;
        dst_col0[12 * rows] = sum12;
        dst_col0[13 * rows] = sum13;
        dst_col0[14 * rows] = sum14;
        dst_col0[15 * rows] = sum15;
        dst_col0[16 * rows] = sum16;
        dst_col0[17 * rows] = sum17;
        dst_col0[18 * rows] = sum18;
        dst_col0[19 * rows] = sum19;
        dst_col0[20 * rows] = sum20;
        dst_col0[21 * rows] = sum21;
        dst_col0[22 * rows] = sum22;
        dst_col0[23 * rows] = sum23;
        dst_col0[24 * rows] = sum24;
        dst_col0[25 * rows] = sum25;
        dst_col0[26 * rows] = sum26;
        dst_col0[27 * rows] = sum27;
        dst_col0[28 * rows] = sum28;
        dst_col0[29 * rows] = sum29;
        dst_col0[30 * rows] = sum30;
        dst_col0[31 * rows] = sum31;
    }
}
