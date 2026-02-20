.. meta::
  :description: llama.cpp examples
  :keywords: llama.cpp, programming, C++, ROCm, example, sample, tutorial, GGML

.. _run-a-llama-cpp-example:

********************************************************************
Run a llama.cpp example
********************************************************************

The `https://github.com/ROCm/llama.cpp <https://github.com/ROCm/llama.cpp>`__ repository provides the necessary examples that exercise the functionality of your
framework.

You can also search for llama.cpp examples on the `AMD ROCm blog <https://rocm.blogs.amd.com/>`_, 
to find instructions to prepare your model and test the containers.

Two most popular use-cases are:

* **llama-cli**: The main executable to run the model interactively or get a response to a prompt.
* **llama-bench**: Run a benchmark of your model with different configurations.

Main Application: ``llama-cli`` 
================================================================================

1. Use the CLI tool to start the client:

   .. code-block:: bash

      ./build/bin/llama-cli -m /data/DeepSeek-V3-Q4_K_M/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf -ngl 999

2. A prompt will appear when the client is ready, and you can start interacting with the model using the client:

   .. code-block::

      > hi, who are you?
      Hi! I’m an AI assistant here to help answer your questions, provide information, or just chat with you.
      How can I assist you today? 😊

      > What are the main causes of heart failure?
      Heart failure is a condition in which the heart cannot pump blood effectively to meet the body's needs.
      It can result from various underlying causes or contributing factors.
      The **main causes of heart failure** include:

      ---

      ### 1. **Coronary Artery Disease (CAD)**
         - Narrowing or blockage of the coronary arteries reduces blood flow to the heart muscle, weakening it over time.
         - A heart attack (myocardial infarction) can cause significant damage to the heart muscle, leading to heart failure.

      ---

      ### 2. **High Blood Pressure (Hypertension)**
         - Chronic high blood pressure forces the heart to work harder to pump blood, eventually causing the heart muscle to thicken or weaken.

      ---

      ### 3. **Cardiomyopathy**
         - Diseases of the heart muscle, such as dilated cardiomyopathy, hypertrophic cardiomyopathy, or restrictive cardiomyopathy, can impair the heart's ability to pump effectively.
      ...
      ### 11. **Other Conditions**
         - Thyroid disorders, severe anemia, or infections like myocarditis can also lead to heart failure.

      ---

      ### Prevention and Management

Benchmark Application: ``llama-bench``
================================================================================

1. Use the CLI tool to start the application:

   .. code-block:: bash

      ./build/bin/llama-bench \
      -m /data/DeepSeek-V3-Q4_K_M/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf \
      -p 16,32,64,96,128,256,512,1024,2048,4096 \
      -n 64,128,256 \
      -ngl 999

2. The result of the command above should be similar to the following when running on a MI300X system:

   .. code-block::

      ggml_cuda_init: GGML_CUDA_FORCE_MMQ:    no
      ggml_cuda_init: GGML_CUDA_FORCE_CUBLAS: no
      ggml_cuda_init: found 8 ROCm devices:
      Device 0: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 1: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 2: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 3: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 4: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 5: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 6: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      Device 7: AMD Instinct MI300X, gfx942:sramecc+:xnack- (0x942), VMM: no, Wave Size: 64
      | model                          |       size |     params | backend    | ngl |            test |                  t/s |
      | ------------------------------ | ---------: | ---------: | ---------- | --: | --------------: | -------------------: |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |            pp16 |        118.10 ± 4.78 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |            pp32 |        153.70 ± 3.11 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |            pp64 |        191.69 ± 1.95 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |            pp96 |        180.58 ± 2.99 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |           pp128 |        192.25 ± 3.14 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |           pp256 |        318.73 ± 3.79 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |           pp512 |        513.29 ± 4.07 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |          pp1024 |        880.58 ± 5.17 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |          pp2048 |       1358.24 ± 2.49 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |          pp4096 |       1650.81 ± 4.47 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |            tg64 |         42.94 ± 0.08 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |           tg128 |         42.24 ± 0.04 |
      | deepseek2 671B Q4_K - Medium   | 376.65 GiB |   671.03 B | ROCm       | 999 |           tg256 |         41.82 ± 0.10 |

      build: 071e9e45 (6662)