message(STATUS "Using HRX backend")

find_package(hrx CONFIG REQUIRED)

ggml_add_backend_library(ggml-hrx
                         ggml-hrx.cpp
                         ../../include/ggml-hrx.h
                        )

target_link_libraries(ggml-hrx PRIVATE hrx::hrx)
