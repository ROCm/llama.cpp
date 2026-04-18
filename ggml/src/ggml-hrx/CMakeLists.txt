message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

if (NOT GGML_HRX_ROCM_PATH)
    if (DEFINED ENV{ROCM_PATH})
        set(GGML_HRX_ROCM_PATH "$ENV{ROCM_PATH}")
    else()
        message(FATAL_ERROR "GGML_HRX_ROCM_PATH or ROCM_PATH is required to build HRX HIP kernels")
    endif()
endif()

if (NOT GGML_HRX_CLANGXX)
    find_program(GGML_HRX_CLANGXX
        NAMES clang++
        HINTS
            "${GGML_HRX_ROCM_PATH}/lib/llvm/bin"
            "${GGML_HRX_ROCM_PATH}/llvm/bin"
    )
endif()

if (NOT GGML_HRX_CLANGXX)
    message(FATAL_ERROR "clang++ is required to build HRX HIP kernels")
endif()

set(GGML_HRX_AMDGPU_TARGETS "gfx1100" CACHE STRING "AMDGPU gfx targets to build HRX HSACO kernels for")
string(REPLACE "," ";" GGML_HRX_AMDGPU_TARGET_LIST "${GGML_HRX_AMDGPU_TARGETS}")

set(GGML_HRX_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(GGML_HRX_HSACO_ROOT "${GGML_HRX_GENERATED_DIR}/hsaco")
set(GGML_HRX_KERNEL_CATALOG_CPP "${GGML_HRX_GENERATED_DIR}/hrx_kernel_catalog.cpp")
set(GGML_HRX_KERNEL_GENERATOR "${CMAKE_CURRENT_SOURCE_DIR}/kernels/generate_hrx_kernels.py")

set(GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
set(GGML_HRX_CATALOG_ARCH_ARGS)
foreach(GGML_HRX_AMDGPU_TARGET IN LISTS GGML_HRX_AMDGPU_TARGET_LIST)
    string(STRIP "${GGML_HRX_AMDGPU_TARGET}" GGML_HRX_AMDGPU_TARGET)
    if (GGML_HRX_AMDGPU_TARGET STREQUAL "")
        continue()
    endif()
    list(APPEND GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL "${GGML_HRX_AMDGPU_TARGET}")
    list(APPEND GGML_HRX_CATALOG_ARCH_ARGS --arch "${GGML_HRX_AMDGPU_TARGET}")
endforeach()

if (NOT GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
    message(FATAL_ERROR "GGML_HRX_AMDGPU_TARGETS must contain at least one target")
endif()

file(MAKE_DIRECTORY "${GGML_HRX_GENERATED_DIR}" "${GGML_HRX_HSACO_ROOT}")
set(GGML_HRX_HSACO_OUTPUTS)

function(ggml_hrx_add_kernel_source)
    cmake_parse_arguments(ARG "" "SOURCE;WAVEFRONT_SIZE" "DEPS;FLAGS" ${ARGN})
    if (NOT ARG_SOURCE)
        message(FATAL_ERROR "ggml_hrx_add_kernel_source requires SOURCE")
    endif()
    if (NOT ARG_WAVEFRONT_SIZE)
        message(FATAL_ERROR "ggml_hrx_add_kernel_source requires WAVEFRONT_SIZE")
    endif()
    if (NOT ARG_WAVEFRONT_SIZE STREQUAL "32" AND NOT ARG_WAVEFRONT_SIZE STREQUAL "64")
        message(FATAL_ERROR "unsupported HRX kernel wavefront size: ${ARG_WAVEFRONT_SIZE}")
    endif()

    get_filename_component(GGML_HRX_KERNEL_SOURCE_NAME "${ARG_SOURCE}" NAME)
    string(REPLACE ".hip.cpp" ".hsaco" GGML_HRX_HSACO_NAME "${GGML_HRX_KERNEL_SOURCE_NAME}")

    set(GGML_HRX_KERNEL_COMPILE_FLAGS)
    if (ARG_WAVEFRONT_SIZE STREQUAL "32")
        list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mno-wavefrontsize64)
    else()
        list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS -mwavefrontsize64)
    endif()
    list(APPEND GGML_HRX_KERNEL_COMPILE_FLAGS ${ARG_FLAGS})

    foreach(GGML_HRX_AMDGPU_TARGET IN LISTS GGML_HRX_AMDGPU_TARGET_LIST_CANONICAL)
        set(GGML_HRX_HSACO_DIR "${GGML_HRX_HSACO_ROOT}/${GGML_HRX_AMDGPU_TARGET}")
        set(GGML_HRX_HSACO_OUTPUT "${GGML_HRX_HSACO_DIR}/${GGML_HRX_HSACO_NAME}")
        file(MAKE_DIRECTORY "${GGML_HRX_HSACO_DIR}")
        add_custom_command(
            OUTPUT "${GGML_HRX_HSACO_OUTPUT}"
            COMMAND "${GGML_HRX_CLANGXX}"
                -x hip
                --offload-device-only
                --no-gpu-bundle-output
                "--offload-arch=${GGML_HRX_AMDGPU_TARGET}"
                -O2
                ${GGML_HRX_KERNEL_COMPILE_FLAGS}
                -c "${ARG_SOURCE}"
                -o "${GGML_HRX_HSACO_OUTPUT}"
            DEPENDS "${ARG_SOURCE}" ${ARG_DEPS}
            VERBATIM
        )
        list(APPEND GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUT}")
    endforeach()
    set(GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUTS}" PARENT_SCOPE)
endfunction()

set(GGML_HRX_WAVE32_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add8_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/argsort_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/clamp_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/concat_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy_f32_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/copy_strided_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/div_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_bf16_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f16_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_f32_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q4_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q8_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/flash_attn_ext_f32_q8_0_q4_0_decode.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gated_delta_net_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/get_rows_q5_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_add_add_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_f32_broadcast.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_bf16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f16_batched.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_f32_batched.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_id_q4_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q4_k_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q8_0.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_sum8_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/quantize_q8_1.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rms_norm.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/rope_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/row_reduce_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/scale_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f16.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_q4_0.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/set_rows_f32_q8_0.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/soft_max_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ssm_conv_update_f32.hip.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/unary_f32.hip.cpp")

set(GGML_HRX_WAVE64_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/ssm_conv_f32.hip.cpp"
)

foreach(GGML_HRX_KERNEL_SOURCE IN LISTS GGML_HRX_WAVE32_KERNEL_SOURCES)
    ggml_hrx_add_kernel_source(SOURCE "${GGML_HRX_KERNEL_SOURCE}" WAVEFRONT_SIZE 32)
endforeach()

foreach(GGML_HRX_KERNEL_SOURCE IN LISTS GGML_HRX_WAVE64_KERNEL_SOURCES)
    ggml_hrx_add_kernel_source(SOURCE "${GGML_HRX_KERNEL_SOURCE}" WAVEFRONT_SIZE 64)
endforeach()

ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q5_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_wave64.hip.cpp"
    WAVEFRONT_SIZE 64
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/mul_mat_vec_q6_k_q8_1_common.hip.inc"
    FLAGS -O3
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_shared.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
)
ggml_hrx_add_kernel_source(
    SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave32.hip.cpp"
    WAVEFRONT_SIZE 32
    DEPS
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_common.hip.inc"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/topk_moe_f32_wave.inc"
)

add_custom_command(
    OUTPUT "${GGML_HRX_KERNEL_CATALOG_CPP}"
    COMMAND "${Python3_EXECUTABLE}"
        "${GGML_HRX_KERNEL_GENERATOR}"
        ${GGML_HRX_CATALOG_ARCH_ARGS}
        --source-dir "${CMAKE_CURRENT_SOURCE_DIR}/kernels"
        --hsaco-root "${GGML_HRX_HSACO_ROOT}"
        --output "${GGML_HRX_KERNEL_CATALOG_CPP}"
    DEPENDS
        "${GGML_HRX_KERNEL_GENERATOR}"
        "${CMAKE_CURRENT_SOURCE_DIR}/kernels/hrx_kernel_catalog.h"
        ${GGML_HRX_HSACO_OUTPUTS}
    VERBATIM
)

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         "${GGML_HRX_KERNEL_CATALOG_CPP}"
                         ../../include/ggml-hrx.h
                         kernels/hrx_kernel_catalog.h
                        )

target_link_libraries(ggml-hrx PRIVATE hrx::hrx)
target_include_directories(ggml-hrx PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}" "${GGML_HRX_GENERATED_DIR}")
