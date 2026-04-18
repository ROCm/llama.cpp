#if !defined(GGML_HRX_FORCE_GENERIC_FA_DIRECT) && \
        (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__))
#include "flash_attn_ext_f32_f16_prefill_direct_gfx11.inc"
#else
#include "flash_attn_ext_f32_f16_prefill_direct_generic.inc"
#endif
