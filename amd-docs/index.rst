.. meta::
  :description: llama.cpp documentation
  :keywords: llama.cpp, ROCm, documentation, GGML, deep learning, framework, GPU

.. _llama-cpp-documentation-index:

********************************************************************
llama.cpp on ROCm documentation
********************************************************************

Run llama.cpp on ROCm to deliver optimized LLM inference on AMD Instinct
GPUs and CPUs, enabling low-latency, memory-efficient on-prem deployments
for chat, summarization, and code assistance.

`llama.cpp <https://github.com/ggml-org/llama.cpp>`__ is an open-source inference library and
framework for Large Language Models (LLMs) that runs on both central processing units 
(CPUs) and graphics processing units (GPUs). It is written in plain C/C++, providing 
a simple, dependency-free setup. 

llama.cpp on ROCm supports multiple quantization options, from 1.5-bit to 8-bit integers, 
to accelerate inference and reduce memory usage. Originally built as a CPU-first library, 
llama.cpp is easy to integrate with other programming environments and is widely 
adopted across diverse platforms, including consumer devices. 

llama.cpp is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-25.08/>`__.

The llama.cpp public repository is located at `https://github.com/ROCm/llama.cpp <https://github.com/ROCm/llama.cpp>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install llama.cpp <install/llama-cpp-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a llama.cpp example <examples/llama-cpp-examples>`

  .. grid-item-card:: Reference

      * `API reference (upstream) <https://llama-cpp-python.readthedocs.io/en/latest/api-reference.html>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.
