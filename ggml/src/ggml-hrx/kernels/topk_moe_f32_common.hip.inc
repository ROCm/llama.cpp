#include <float.h>
#include <stdint.h>

struct hrx_topk_moe_f32_constants {
    long long n_experts;
    long long n_rows;
    long long n_expert_used;
    long long logits_nb1;
    long long weights_nb1;
    long long weights_nb_k;
    long long ids_nb1;
    long long ids_nb_k;
    float scale;
    float clamp_min;
    float clamp_max;
    int with_norm;
};

static __device__ __forceinline__ bool hrx_topk_take_rhs(
        float lhs_v, int lhs_i, float rhs_v, int rhs_i) {
    return rhs_v > lhs_v || (rhs_v == lhs_v && rhs_i < lhs_i);
}

static __device__ __forceinline__ float hrx_topk_clamp_denom(
        float value, const hrx_topk_moe_f32_constants & c) {
    return c.with_norm ?
        (value < c.clamp_min ? c.clamp_min : (value > c.clamp_max ? c.clamp_max : value)) :
        value;
}
