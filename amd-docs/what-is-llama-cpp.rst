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

llama.cpp can be applied in a variety of scenarios, particularly when you need to meet one or more of the following requirements:

- Plain C/C++ implementation with no external dependencies
- Support for 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory usage
- Custom HIP (Heterogeneous-compute Interface for Portability) kernels for running large language models (LLMs) on AMD GPUs (graphics processing units)
- CPU (central processing unit) + GPU (graphics processing unit) hybrid inference for partially accelerating models larger than the total available VRAM (video random-access memory)

llama.cpp is also used in a range of real-world applications, including:

- Games such as `Lucy's Labyrinth <https://github.com/MorganRO8/Lucys_Labyrinth>`__:
  A simple maze game where AI-controlled agents attempt to trick the player.
- Tools such as `Styled Lines <https://marketplace.unity.com/packages/tools/ai-ml-integration/style-text-webgl-ios-stand-alone-llm-llama-cpp-wrapper-292902>`__:
  A proprietary, asynchronous inference wrapper for Unity3D game development, including pre-built mobile and web platform wrappers and a model example.
- Various other AI applications use llama.cpp as their inference engine;  
  for a detailed list, see the `user interfaces (UIs) section <https://github.com/ggml-org/llama.cpp?tab=readme-ov-file#description>`__.

For more use cases and recommendations, refer to the `AMD ROCm blog <https://rocm.blogs.amd.com/>`__, 
where you can search for llama.cpp examples and best practices to optimize your workloads on AMD GPUs.

- The `Llama.cpp Meets Instinct: A New Era of Open-Source AI Acceleration <https://rocm.blogs.amd.com/ecosystems-and-partners/llama-cpp/README.html>`__ 
  blog post outlines how the open-source llama.cpp framework enables efficient LLM inference—including interactive inference with ``llama-cli``, 
  server deployment with ``llama-server``, GGUF model preparation and quantization, performance benchmarking, and optimizations tailored for 
  AMD Instinct GPUs within the ROCm ecosystem. 

