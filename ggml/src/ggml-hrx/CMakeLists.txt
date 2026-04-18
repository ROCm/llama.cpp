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
set(GGML_HRX_KERNEL_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/kernels/add_f32.hip.cpp")
file(MAKE_DIRECTORY "${GGML_HRX_GENERATED_DIR}" "${GGML_HRX_HSACO_ROOT}")

set(GGML_HRX_HSACO_OUTPUTS)
foreach(GGML_HRX_AMDGPU_TARGET IN LISTS GGML_HRX_AMDGPU_TARGET_LIST)
    string(STRIP "${GGML_HRX_AMDGPU_TARGET}" GGML_HRX_AMDGPU_TARGET)
    if (GGML_HRX_AMDGPU_TARGET STREQUAL "")
        continue()
    endif()

    set(GGML_HRX_HSACO_DIR "${GGML_HRX_HSACO_ROOT}/${GGML_HRX_AMDGPU_TARGET}")
    set(GGML_HRX_HSACO_OUTPUT "${GGML_HRX_HSACO_DIR}/add_f32.hsaco")
    file(MAKE_DIRECTORY "${GGML_HRX_HSACO_DIR}")
    add_custom_command(
        OUTPUT "${GGML_HRX_HSACO_OUTPUT}"
        COMMAND "${GGML_HRX_CLANGXX}"
            -x hip
            --offload-device-only
            --no-gpu-bundle-output
            "--offload-arch=${GGML_HRX_AMDGPU_TARGET}"
            -O2
            -c "${GGML_HRX_KERNEL_SOURCE}"
            -o "${GGML_HRX_HSACO_OUTPUT}"
        DEPENDS "${GGML_HRX_KERNEL_SOURCE}"
        VERBATIM
    )
    list(APPEND GGML_HRX_HSACO_OUTPUTS "${GGML_HRX_HSACO_OUTPUT}")
    list(APPEND GGML_HRX_CATALOG_ARCH_ARGS --arch "${GGML_HRX_AMDGPU_TARGET}")
endforeach()

if (NOT GGML_HRX_HSACO_OUTPUTS)
    message(FATAL_ERROR "GGML_HRX_AMDGPU_TARGETS must contain at least one target")
endif()

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
