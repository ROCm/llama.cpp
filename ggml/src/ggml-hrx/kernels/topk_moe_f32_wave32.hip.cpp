#include <hip/hip_runtime.h>

#include "topk_moe_f32_common.hip.inc"

#define HRX_TOPK_MOE_WAVE_KERNEL hrx_topk_moe_f32_wave32
#define HRX_TOPK_MOE_WAVE_SIZE 32
#include "topk_moe_f32_wave.inc"
