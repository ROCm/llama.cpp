static __device__ __forceinline__ int hrx_topk_shfl_xor_i32(
        int value, int lane, int mask) {
    return __builtin_amdgcn_ds_bpermute((lane ^ mask) << 2, value);
}

static __device__ __forceinline__ float hrx_topk_shfl_xor_f32(
        float value, int lane, int mask) {
    union {
        int i;
        float f;
    } tmp;
    tmp.f = value;
    tmp.i = hrx_topk_shfl_xor_i32(tmp.i, lane, mask);
    return tmp.f;
}

extern "C" __global__ void HRX_TOPK_MOE_WAVE_KERNEL(
        const float * logits, float * weights, int * ids,
        hrx_topk_moe_f32_constants c) {
    constexpr int wave_size = HRX_TOPK_MOE_WAVE_SIZE;
    constexpr int experts_per_lane = 256 / wave_size;

    float probs[experts_per_lane];
    float selection_probs[experts_per_lane];
    float output_weights[experts_per_lane];

    const int lane = static_cast<int>(__builtin_amdgcn_workitem_id_x());
    const int row_in_group = static_cast<int>(__builtin_amdgcn_workitem_id_y());
    const long long row = static_cast<long long>(__builtin_amdgcn_workgroup_id_x()) *
        static_cast<long long>(__builtin_amdgcn_workgroup_size_y()) + row_in_group;
    if (row >= c.n_rows) {
        return;
    }

    const float * logits_row = reinterpret_cast<const float *>(
        reinterpret_cast<const char *>(logits) + row * c.logits_nb1);
    char * weights_row = reinterpret_cast<char *>(weights) + row * c.weights_nb1;
    char * ids_row = reinterpret_cast<char *>(ids) + row * c.ids_nb1;

    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        const int expert = lane + i * wave_size;
        probs[i] = expert < c.n_experts ? logits_row[expert] * c.scale : -FLT_MAX;
    }

    float max_val = probs[0];
    #pragma unroll
    for (int i = 1; i < experts_per_lane; ++i) {
        max_val = fmaxf(max_val, probs[i]);
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        max_val = fmaxf(max_val, hrx_topk_shfl_xor_f32(max_val, lane, mask));
    }

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        const int expert = lane + i * wave_size;
        probs[i] = expert < c.n_experts ? __expf(probs[i] - max_val) : 0.0f;
        sum += probs[i];
    }
    #pragma unroll
    for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
        sum += hrx_topk_shfl_xor_f32(sum, lane, mask);
    }

    const float inv_sum = 1.0f / sum;
    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        const int expert = lane + i * wave_size;
        probs[i] = expert < c.n_experts ? probs[i] * inv_sum : -FLT_MAX;
        selection_probs[i] = probs[i];
        output_weights[i] = 0.0f;
    }

    float selected_sum = 0.0f;
    for (int k = 0; k < c.n_expert_used; ++k) {
        float best_v = probs[0];
        float best_s = selection_probs[0];
        int best_i = lane;
        #pragma unroll
        for (int i = 1; i < experts_per_lane; ++i) {
            const int expert = lane + i * wave_size;
            if (expert < c.n_experts && hrx_topk_take_rhs(best_s, best_i, selection_probs[i], expert)) {
                best_v = probs[i];
                best_s = selection_probs[i];
                best_i = expert;
            }
        }

        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            const float rhs_v = hrx_topk_shfl_xor_f32(best_v, lane, mask);
            const float rhs_s = hrx_topk_shfl_xor_f32(best_s, lane, mask);
            const int rhs_i = hrx_topk_shfl_xor_i32(best_i, lane, mask);
            if (hrx_topk_take_rhs(best_s, best_i, rhs_s, rhs_i)) {
                best_v = rhs_v;
                best_s = rhs_s;
                best_i = rhs_i;
            }
        }

        if ((k & (wave_size - 1)) == lane) {
            output_weights[k / wave_size] = best_v;
        }
        if ((best_i & (wave_size - 1)) == lane) {
            selection_probs[best_i / wave_size] = -FLT_MAX;
            *reinterpret_cast<int *>(ids_row + k * c.ids_nb_k) = best_i;
            selected_sum += best_v;
        }
    }

    if (c.with_norm) {
        #pragma unroll
        for (int mask = wave_size / 2; mask > 0; mask >>= 1) {
            selected_sum += hrx_topk_shfl_xor_f32(selected_sum, lane, mask);
        }
        const float denom = hrx_topk_clamp_denom(selected_sum, c);
        #pragma unroll
        for (int i = 0; i < experts_per_lane; ++i) {
            output_weights[i] /= denom;
        }
    }

    #pragma unroll
    for (int i = 0; i < experts_per_lane; ++i) {
        const int k = i * wave_size + lane;
        if (k < c.n_expert_used) {
            *reinterpret_cast<float *>(weights_row + k * c.weights_nb_k) = output_weights[i];
        }
    }
}
