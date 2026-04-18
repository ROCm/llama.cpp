#!/usr/bin/env python3
import argparse
import pathlib
import re
import sys


KERNELS = [
    {
        "name": "hrx_rms_norm_f32",
        "source": "rms_norm.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (512, 1, 1),
    },
    {
        "name": "hrx_add_f32",
        "source": "add_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_f32",
        "source": "mul_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add_f32_broadcast",
        "source": "add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_f32_broadcast",
        "source": "mul_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_div_f32_broadcast",
        "source": "div_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_add8_f32",
        "source": "add8_f32.hip.cpp",
        "format": None,
        "binding_count": 9,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_sum8_f32",
        "source": "mul_sum8_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 56,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_add_add_f32_broadcast",
        "source": "mul_add_add_f32_broadcast.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 176,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_scale_f32",
        "source": "scale_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "parameter_count": 5,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_f32",
        "source": "set_rows_f32_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_f16",
        "source": "set_rows_f32_f16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_q8_0",
        "source": "set_rows_f32_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_set_rows_f32_q4_0",
        "source": "set_rows_f32_q4_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_silu_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sigmoid_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_softplus_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_swiglu_f32",
        "source": "unary_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_sum_rows_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_l2_norm_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_l2_norm_wg128_f32",
        "source": "row_reduce_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_soft_max_f32",
        "source": "soft_max_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_soft_max_f32_mask",
        "source": "soft_max_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_argsort_f32_i32",
        "source": "argsort_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_rope_f32",
        "source": "rope_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_ssm_conv_f32",
        "source": "ssm_conv_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 88,
        "workgroup_size": (32, 16, 1),
    },
    {
        "name": "hrx_ssm_conv_update_f32",
        "source": "ssm_conv_update_f32.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_gated_delta_net_f32",
        "source": "gated_delta_net_f32.hip.cpp",
        "format": None,
        "binding_count": 8,
        "constants_size": 208,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_clamp_f32",
        "source": "clamp_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 16,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_f32",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_f32_nr1",
        "source": "get_rows_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_get_rows_q5_k_f32",
        "source": "get_rows_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wg128_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wg64_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols1_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols1_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols1_wg32_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows4_k512_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows4_k2048_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols4_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols8_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_cols32_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_rows2_cols16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_wmma16x16_f32",
        "source": "mul_mat_vec_bf16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wg128_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wg64_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_wmma16x16_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols1_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows2_cols1_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows4_k2048_cols1_lds_wg256_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols4_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols8_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_cols16_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_swiglu_rows2_cols8_f32",
        "source": "mul_mat_vec_bf16_swiglu.hip.cpp",
        "format": None,
        "binding_count": 4,
        "parameter_count": 7,
        "constants_size": 24,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_bf16_set_rows_f16",
        "source": "mul_mat_vec_bf16_set_rows.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 40,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_f32",
        "source": "mul_mat_vec_f16.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols1_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols4_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols8_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f16_batched_cols16_f32",
        "source": "mul_mat_vec_f16_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_f32",
        "source": "mul_mat_vec_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols1_ne2_1_k2048_wg32_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols8_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_cols16_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_f32_batched_rows2_cols8_f32",
        "source": "mul_mat_vec_f32_batched.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 128,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_row4_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 104,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_clear_u32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 1,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_compact_moe_routes_i32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 48,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_grouped_q8_1_x4_mmq64x64_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 96,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_q8_1_f32",
        "source": "mul_mat_id_q4_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 4,
        "constants_size": 96,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_rows2_x16_wg32_f32",
        "source": "mul_mat_id_q4_k_mul.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 112,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_row4_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_row2_route8_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_grouped_q8_1_x4_mmq32x64_wg64_f32",
        "source": "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 112,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_swiglu_packed_wg64_f32",
        "source": "mul_mat_id_q4_k_swiglu.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 120,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_mul_mat_id_q4_k_mul_q8_1_f32",
        "source": "mul_mat_id_q4_k_mul_q8_1.hip.cpp",
        "format": None,
        "binding_count": 5,
        "constants_size": 104,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q4_k_f32",
        "source": "mul_mat_vec_q4_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_quantize_q8_1_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (32, 1, 1),
    },
    {
        "name": "hrx_quantize_q8_1_x4_f32",
        "source": "quantize_q8_1.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 56,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q4_k_q8_1_f32",
        "source": "mul_mat_vec_q4_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_f32",
        "source": "mul_mat_vec_q5_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q5_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmql128x128_wg256_f32",
        "source": "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q5_k_q8_1_x4_mmq64x64_wg256_f32",
        "source": "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_f32",
        "source": "mul_mat_vec_q6_k.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmql128x64_wg256_f32",
        "source": "mul_mat_vec_q6_k_q8_1_wave64.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q6_k_q8_1_x4_mmq32x32_wg128_f32",
        "source": "mul_mat_vec_q6_k_q8_1.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (128, 1, 1),
    },
    {
        "name": "hrx_mul_mat_vec_q8_0_f32",
        "source": "mul_mat_vec_q8_0.hip.cpp",
        "format": None,
        "binding_count": 3,
        "parameter_count": 6,
        "constants_size": 24,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_decode",
        "source": "flash_attn_ext_f32_f16_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_tile8",
        "source": "flash_attn_ext_f32_f16_prefill_tile8.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_wmma16",
        "source": "flash_attn_ext_f32_f16_prefill_wmma16.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f16_prefill_direct",
        "source": "flash_attn_ext_f32_f16_prefill_direct.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
        "arch_prefixes": ("gfx11",),
    },
    {
        "name": "hrx_flash_attn_ext_f32_bf16_decode",
        "source": "flash_attn_ext_f32_bf16_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_f32_decode",
        "source": "flash_attn_ext_f32_f32_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q8_0_decode",
        "source": "flash_attn_ext_f32_q8_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q4_0_decode",
        "source": "flash_attn_ext_f32_q4_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_flash_attn_ext_f32_q8_0_q4_0_decode",
        "source": "flash_attn_ext_f32_q8_0_q4_0_decode.hip.cpp",
        "format": None,
        "binding_count": 6,
        "constants_size": 200,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_topk_moe_f32",
        "source": "topk_moe_f32_shared.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (64, 1, 1),
    },
    {
        "name": "hrx_topk_moe_f32_shared4",
        "source": "topk_moe_f32_shared.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (64, 4, 1),
    },
    {
        "name": "hrx_topk_moe_f32_wave32",
        "source": "topk_moe_f32_wave32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 80,
        "workgroup_size": (32, 4, 1),
    },
    {
        "name": "hrx_concat_f32",
        "source": "concat_f32.hip.cpp",
        "format": None,
        "binding_count": 3,
        "constants_size": 72,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_copy_strided_f32",
        "source": "copy_strided_f32.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 64,
        "workgroup_size": (256, 1, 1),
    },
    {
        "name": "hrx_copy_f32_f16",
        "source": "copy_f32_f16.hip.cpp",
        "format": None,
        "binding_count": 2,
        "constants_size": 8,
        "workgroup_size": (256, 1, 1),
    },
]

DOT8_ARCH_PREFIXES = ("gfx11",)
DOT8_SOURCES = {
    "mul_mat_id_q4_k_q8_1_x4_mmq.hip.cpp",
    "mul_mat_vec_q5_k_q8_1.hip.cpp",
    "mul_mat_vec_q5_k_q8_1_wave64.hip.cpp",
    "mul_mat_vec_q6_k_q8_1.hip.cpp",
    "mul_mat_vec_q6_k_q8_1_wave64.hip.cpp",
    "mul_mat_vec_q8_0.hip.cpp",
}


def hsaco_name_for_source(source):
    if source.endswith(".hip.cpp"):
        return source[:-len(".hip.cpp")] + ".hsaco"
    return pathlib.Path(source).name + ".hsaco"


def symbol_part(value):
    return re.sub(r"[^A-Za-z0-9_]", "_", value)


def unique_sources():
    return sorted({kernel["source"] for kernel in KERNELS})


def kernel_supports_arch(kernel, arch):
    prefixes = DOT8_ARCH_PREFIXES if kernel["source"] in DOT8_SOURCES else kernel.get("arch_prefixes")
    return not prefixes or any(arch.startswith(prefix) for prefix in prefixes)


def write_catalog(output, entries):
    with output.open("w", encoding="utf-8") as f:
        f.write("// Generated by generate_hrx_kernels.py. Do not edit.\n")
        f.write('#include "kernels/hrx_kernel_catalog.h"\n')
        f.write("#include <cstring>\n\n")

        written_symbols = set()
        for entry in entries:
            if entry["data_symbol"] in written_symbols:
                continue
            written_symbols.add(entry["data_symbol"])
            data = entry["data"]
            f.write(f"alignas(16) static const unsigned char {entry['data_symbol']}[] = {{\n")
            for i in range(0, len(data), 12):
                chunk = data[i:i + 12]
                f.write("    ")
                f.write(", ".join(f"0x{value:02x}" for value in chunk))
                f.write(",\n")
            f.write("};\n\n")

        f.write("static const ggml_hrx_kernel_entry k_hrx_kernel_catalog[] = {\n")
        for entry in entries:
            wx, wy, wz = entry["workgroup_size"]
            executable_format = entry["format"]
            executable_format_c = "nullptr" if not executable_format else f'"{executable_format}"'
            f.write("    {\n")
            f.write(f'        "{entry["name"]}",\n')
            f.write(f'        "{entry["gfx_target"]}",\n')
            f.write(f"        {entry['data_symbol']},\n")
            f.write(f"        sizeof({entry['data_symbol']}),\n")
            f.write(f"        {executable_format_c},\n")
            f.write(f"        {entry['binding_count']},\n")
            f.write(f"        {entry['parameter_count']},\n")
            f.write(f"        {entry['constants_size']},\n")
            f.write(f"        {{ {wx}, {wy}, {wz} }},\n")
            f.write("    },\n")
        f.write("};\n\n")

        f.write("const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_entries(size_t * count) {\n")
        f.write("    if (count) {\n")
        f.write("        *count = sizeof(k_hrx_kernel_catalog) / sizeof(k_hrx_kernel_catalog[0]);\n")
        f.write("    }\n")
        f.write("    return k_hrx_kernel_catalog;\n")
        f.write("}\n\n")

        f.write("const ggml_hrx_kernel_entry * ggml_hrx_kernel_catalog_find(const char * name, const char * gfx_target) {\n")
        f.write("    if (!name || !gfx_target) {\n")
        f.write("        return nullptr;\n")
        f.write("    }\n")
        f.write("    size_t count = 0;\n")
        f.write("    const ggml_hrx_kernel_entry * entries = ggml_hrx_kernel_catalog_entries(&count);\n")
        f.write("    for (size_t i = 0; i < count; ++i) {\n")
        f.write("        if (std::strcmp(entries[i].name, name) == 0 && std::strcmp(entries[i].gfx_target, gfx_target) == 0) {\n")
        f.write("            return &entries[i];\n")
        f.write("        }\n")
        f.write("    }\n")
        f.write("    return nullptr;\n")
        f.write("}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", action="append", default=[])
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--hsaco-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--list-sources", action="store_true")
    args = parser.parse_args()

    source_dir = pathlib.Path(args.source_dir)
    if args.list_sources:
        for source in unique_sources():
            print(source_dir / source)
        return 0

    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if not args.arch:
        sys.stderr.write("error: at least one --arch is required\n")
        return 1
    if not args.hsaco_root:
        sys.stderr.write("error: --hsaco-root is required\n")
        return 1

    hsaco_root = pathlib.Path(args.hsaco_root)
    entries = []
    hsaco_cache = {}
    for arch in args.arch:
        for kernel in KERNELS:
            if not kernel_supports_arch(kernel, arch):
                continue
            source = kernel["source"]
            key = (arch, source)
            if key not in hsaco_cache:
                hsaco = hsaco_root / arch / hsaco_name_for_source(source)
                if not hsaco.exists():
                    sys.stderr.write(f"error: expected compiled HRX kernel object not found: {hsaco}\n")
                    return 1
                hsaco_cache[key] = {
                    "symbol": f"k_hrx_kernel_{symbol_part(arch)}_{len(hsaco_cache)}",
                    "data": hsaco.read_bytes(),
                }

            entry = dict(kernel)
            entry["gfx_target"] = arch
            entry["parameter_count"] = kernel.get("parameter_count", kernel["binding_count"] + 1)
            entry["data_symbol"] = hsaco_cache[key]["symbol"]
            entry["data"] = hsaco_cache[key]["data"]
            entries.append(entry)

    write_catalog(output, entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
