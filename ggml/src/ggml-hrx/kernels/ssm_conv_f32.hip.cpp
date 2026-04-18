#include <hip/hip_runtime.h>
#include <stdint.h>

struct hrx_ssm_conv_constants {
    long long d_conv;
    long long conv_width;
    long long d_inner;
    long long n_tokens;
    long long n_seqs;
    long long src0_nb1;
    long long src0_nb2;
    long long weight_nb1;
    long long dst_nb1;
    long long dst_nb2;
    int apply_silu;
    int pad;
};

extern "C" __global__ void hrx_ssm_conv_f32(
        const float * src0, const float * weight, float * dst,
        hrx_ssm_conv_constants c) {
    const long long channel = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) * 32 +
        __builtin_amdgcn_workitem_id_x();
    const long long token = static_cast<long long>(__builtin_amdgcn_workgroup_id_y()) * 16 +
        __builtin_amdgcn_workitem_id_y();
    const long long seq = __builtin_amdgcn_workgroup_id_z();
    if (channel >= c.d_inner || token >= c.n_tokens || seq >= c.n_seqs) {
        return;
    }

    const char * src_base = reinterpret_cast<const char *>(src0) +
        channel * c.src0_nb1 + token * sizeof(float) + seq * c.src0_nb2;
    const char * weight_base = reinterpret_cast<const char *>(weight) + channel * c.weight_nb1;
    float sum = 0.0f;
    if (c.d_conv == 4) {
        const float4 x = *reinterpret_cast<const float4 *>(src_base);
        const float4 w = *reinterpret_cast<const float4 *>(weight_base);
        sum = x.x * w.x + x.y * w.y + x.z * w.z + x.w * w.w;
    } else {
        for (long long i = 0; i < c.d_conv; ++i) {
            const float x = *reinterpret_cast<const float *>(src_base + i * sizeof(float));
            const float w = *reinterpret_cast<const float *>(weight_base + i * sizeof(float));
            sum += x * w;
        }
    }
    if (c.apply_silu) {
        sum = sum / (1.0f + __builtin_expf(-sum));
    }

    *reinterpret_cast<float *>(
        reinterpret_cast<char *>(dst) + channel * sizeof(float) + token * c.dst_nb1 + seq * c.dst_nb2) = sum;
}
