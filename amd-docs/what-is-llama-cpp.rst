.. meta::
  :description: What is llama.cpp?
  :keywords: llama.cpp, documentation, GGML, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-llama-cpp:

********************************************************************
What is llama.cpp?
********************************************************************

`llama.cpp <https://github.com/ggml-org/llama.cpp>`__ is an open-source framework 
for Large Language Model (LLM) inference that runs on both central processing units 
(CPUs) and graphics processing units (GPUs). It is written in plain C/C++, providing 
a simple, dependency-free setup. 

The framework supports multiple quantization options, from 1.5-bit to 8-bit integers, 
to accelerate inference and reduce memory usage. Originally built as a CPU-first library, 
llama.cpp is easy to integrate with other programming environments and is widely 
adopted across diverse platforms, including consumer devices. 

Features and use cases
====================================================================

llama.cpp provides the following key features:

- **Portable, Lightweight Inference:** Runs LLMs on CPUs and GPUs with
  minimal dependencies, with HIP/ROCm support for AMD Instinct GPUs.

- **Quantization and GGUF Format:** Supports multiple quantization schemes
  and the GGUF model format to reduce memory usage and enable efficient
  on-device inference.

- **GPU Offload and KV Cache Optimizations:** Offloads attention and other
  compute to GPUs when available and manages KV cache efficiently for
  long-context decoding.

- **Flexible Interfaces:** Offers CLI, server modes, and bindings for
  common languages to integrate into applications and pipelines.

- **Cross-Platform Performance:** Utilizes vectorized kernels and threading
  to achieve strong CPU performance, with optional GPU acceleration on ROCm.

llama.cpp is commonly used in the following scenarios:

- **On-Prem and Edge Inference:** Deploy private LLMs without external
  dependencies on local hardware.

- **Low-Latency Applications:** Power chat, summarization, and code assistance
  with optimized decoding and quantized models.

- **Prototyping and Experimentation:** Quickly evaluate models and prompt
  strategies on diverse hardware setups.

- **Embedded and Resource-Constrained Environments:** Run compact models
  where memory and compute are limited.

llama.cpp can also be used particularly when you need to meet one or more of the following requirements:

1. Plain C/C++ implementation with no external dependencies for simple builds
   and portable deployment.

2. Support for 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer
   quantization enabling faster inference and reduced memory usage.

3. Custom `HIP <https://rocm.docs.amd.com/projects/HIP/en/latest/index.html>`__ kernels for
   running LLMs on AMD GPUs with ROCm to accelerate attention and related operations.

4. CPU and GPU hybrid inference to partially accelerate models larger than the
   total available VRAM by offloading selective computations to the GPU.

llama.cpp is also used in a range of real-world applications, including:

- Games such as `Lucy's Labyrinth <https://github.com/MorganRO8/Lucys_Labyrinth>`__:
  A simple maze game where AI-controlled agents attempt to trick the player.

- Tools such as `Styled Lines <https://marketplace.unity.com/packages/tools/ai-ml-integration/style-text-webgl-ios-stand-alone-llm-llama-cpp-wrapper-292902>`__:
  A proprietary, asynchronous inference wrapper for Unity3D game development, including prebuilt mobile and web platform wrappers and a model example.

- Various other AI applications use llama.cpp as their inference engine.
  For a detailed list, see the `user interfaces (UIs) section <https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#description>`__.

For more use cases and recommendations, refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/>`__, 
where you can search for llama.cpp examples and best practices to optimize your workloads on AMD GPUs.

Why llama.cpp?
====================================================================

llama.cpp is well suited for practical LLM inference for the following reasons:

- Its **lightweight design and quantization support** enables small memory
  footprints and fast startup.

- **GPU offload with ROCm** provides additional throughput on AMD Instinct
  GPUs while maintaining portability.

- **Simple tooling and interfaces** make it easy to integrate into apps,
  services, and batch pipelines.

- **Active community and rapid updates** keep performance and feature
  support evolving across platforms.
  