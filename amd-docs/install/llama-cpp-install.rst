.. meta::
  :description: installing llama.cpp for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, llama.cpp, GGML

.. _llama-cpp-on-rocm-installation:

********************************************************************
llama.cpp on ROCm installation
********************************************************************

System requirements
====================================================================

To use llama.cpp `b6652 <https://github.com/ROCm/llama.cpp/tree/release/b6652>`__, you need the following prerequisites:

- **ROCm version:** `7.0.0 <https://rocm.docs.amd.com/en/docs-7.0.0/>`__
- **Operating system:** Ubuntu 24.04, 22.04
- **GPU platform:** AMD Instinct™ MI325X, MI300X, MI210

Key ROCm libraries for llama.cpp
--------------------------------------------------------------------------------

llama.cpp functionality on ROCm is determined by its underlying library
dependencies. These ROCm components affect the capabilities, performance, and
feature set available to developers. Ensure you have the required libraries for 
your corresponding ROCm version.

.. list-table::
    :header-rows: 1

    * - ROCm library
      - ROCm 7.0.0 version
      - Purpose
      - Usage
    * - `hipBLAS <https://github.com/ROCm/hipBLAS>`__
      - 3.0.0
      - Provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS) for
        matrix and vector operations.
      - Supports operations such as matrix multiplication, matrix-vector
        products, and tensor contractions. Utilized in both dense and batched
        linear algebra operations.
    * - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`__
      - 1.0.0
      - hipBLASLt is an extension of the hipBLAS library, providing additional
        features like epilogues fused into the matrix multiplication kernel or
        use of integer tensor cores.
      - By setting the flag ``ROCBLAS_USE_HIPBLASLT``, you can dispatch hipBLASLt
        kernels where possible.
    * - `rocWMMA <https://github.com/ROCm/rocWMMA>`__
      - 2.0.0
      - Accelerates warp-level matrix-multiply and matrix-accumulate to speed up matrix
        multiplication (GEMM) and accumulation operations with mixed precision
        support.
      - Can be used to enhance the flash attention performance on AMD compute, by enabling
        the flag at compile time.

Install llama.cpp
================================================================================

To install llama.cpp on ROCm, you have the following options:

- :ref:`Use the prebuilt Docker image <use-docker-with-llama-cpp-pre-installed>` **(recommended)**
- :ref:`Build your own Docker image <build-llama-cpp-docker-image>`

.. _use-docker-with-llama-cpp-pre-installed:

Use a prebuilt Docker image with llama.cpp pre-installed
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a llama.cpp environment, and it avoids 
potential installation issues. The tested, prebuilt image includes llama.cpp, ROCm, 
and other dependencies.

.. important::

   To follow these instructions, input your chosen tag into ``<TAG>``. Example: ``llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04``.

   Tag endings of ``_full``, ``_server``, and ``_light`` serve different purposes for entrypoints as follows:

   - Full: This image includes both the main executable file and the tools to convert ``LLaMA`` models into ``ggml`` and apply 4-bit quantization.
   - Server: This image only includes the server executable file.
   - Light: This image only includes the main executable file.

   You can download Docker images with specific ROCm, llama.cpp, and operating system versions. 
   See the available tags on `Docker Hub <https://hub.docker.com/r/rocm/llama.cpp/tags>`_ and see :ref:`Docker image support <llama-cpp-docker-support>` below.

1. Download your required public `llama.cpp Docker image <https://hub.docker.com/r/rocm/llama.cpp/tags>`__:

   .. code-block:: bash

      docker pull rocm/llama.cpp:<TAG>_full
      docker pull rocm/llama.cpp:<TAG>_server
      docker pull rocm/llama.cpp:<TAG>_light

2. Launch and connect to the container with the respective entrypoints of your image:

   .. code-block:: bash

      export MODEL_PATH='<your_model_path>'

      # Multi-GPU Setup (for example, an 8-GPU configuration) is required to load DeepSeek-V3-Q4_K_M model and prevent out-of-memory errors
      # Loading the model may take several minutes depending on the hardware configuration

      # To run the 'full' docker image with main executable (--run) and other options
      docker run --privileged \
                 --network=host \
                 --device=/dev/kfd \
                 --device=/dev/dri \
                 --group-add video \
                 --cap-add=SYS_PTRACE \
                 --security-opt seccomp=unconfined \
                 --ipc=host \
                 --shm-size 16G \
                 -v $MODEL_PATH:/data \
                 rocm/llama.cpp:<TAG>_full \
                   --run -m /data/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf \
                   -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 999

      # To run the 'server' docker image with the server executable
      docker run --privileged \
                 --network=host \
                 --device=/dev/kfd \
                 --device=/dev/dri \
                 --group-add video \
                 --cap-add=SYS_PTRACE \
                 --security-opt seccomp=unconfined \
                 --ipc=host \
                 --shm-size 16G \
                 -v $MODEL_PATH:/data \
                 rocm/llama.cpp:<TAG>_server \
                   -m /data/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf \
                   --port 8000 --host 0.0.0.0 -n 512 --n-gpu-layers 999

      # To run the 'light' docker image with only the main executable
      docker run --privileged \
                 --network=host \
                 --device=/dev/kfd \
                 --device=/dev/dri \
                 --group-add video \
                 --cap-add=SYS_PTRACE \
                 --security-opt seccomp=unconfined \
                 --ipc=host \
                 --shm-size 16G \
                 -v $MODEL_PATH:/data \
                 rocm/llama.cpp:<TAG>_light \
                   -m /data/DeepSeek-V3-Q4_K_M-00001-of-00009.gguf \
                   -p "Building a website can be done in 10 simple steps:" -n 512 --n-gpu-layers 999

   .. note::

       This step will automatically download the image if it does not exist on the host. You can
       also pass the ``-v`` argument to mount any data directories from the host onto the container.

.. _llama-cpp-docker-support:

Docker image support
--------------------------------------------------------------------------------------

AMD validates and publishes ready-made `llama.cpp <https://hub.docker.com/r/rocm/llama.cpp/tags>`__ 
images with ROCm backends on Docker Hub. The following Docker image tags and associated inventories 
are validated for `ROCm 7.0.0 <https://rocm.docs.amd.com/en/docs-7.0.0/>`__.

.. tab-set::

   .. tab-item:: ROCm 7.0.0 - Ubuntu 24.04

      .. tab-set::

         .. tab-item:: Full Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_full <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_full/images/sha256-a94f0c7a598cc6504ff9e8371c016d7a2f93e69bf54a36c870f9522567201f10>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__

         .. tab-item:: Server Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_server <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_server/images/sha256-be175932c3c96e882dfbc7e20e0e834f58c89c2925f48b222837ee929dfc47ee>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__

         .. tab-item:: Light Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_light <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu24.04_light/images/sha256-d8ba0c70603da502c879b1f8010b439c8e7fa9f6cbdac8bbbbbba97cb41ebc9e>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__

   .. tab-item:: ROCm 7.0.0 - Ubuntu 22.04

      .. tab-set::

         .. tab-item:: Full Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_full <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_full/images/sha256-37582168984f25dce636cc7288298e06d94472ea35f65346b3541e6422b678ee>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__

         .. tab-item:: Server Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_server <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_server/images/sha256-7e70578e6c3530c6591cc2c26da24a9ee68a20d318e12241de93c83224f83720>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__

         .. tab-item:: Light Docker

            Tag
              `rocm/llama.cpp:llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_light <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b6652.amd0_rocm7.0.0_ubuntu22.04_light/images/sha256-9a5231acf88b4a229677bc2c636ea3fe78a7a80f558bd80910b919855de93ad5>`__

            Inventory
              * `ROCm 7.0.0 <https://repo.radeon.com/rocm/apt/7.0/>`__


.. _build-llama-cpp-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------------

If you want to explore llama.cpp capabilities without being limited to the entrypoints
from the prebuilt Docker images, you have the option to build directly from source inside a
ROCm Ubuntu base Docker image.

The prebuilt base Docker image has all dependencies installed, including:

* ROCm
* hipBLAS
* hipBLASLt
* rocWMMA

1. Choose your base Ubuntu Docker image with the correct ROCm version.

   .. list-table::
     :header-rows: 1
     :widths: 20 80

     * - Ubuntu Version
       - Base Image
     * - 24.04
       - ``rocm/dev-ubuntu-24.04:7.0-complete``
     * - 22.04
       - ``rocm/dev-ubuntu-22.04:7.0-complete``

2. Start your local container from the base image. The following example uses ``rocm/dev-ubuntu-24.04:7.0-complete``:

   .. code-block:: bash

      export MODEL_PATH='./models'

      docker run -it \
            --name=$(whoami)_llamacpp \
            --privileged --network=host \
            --device=/dev/kfd --device=/dev/dri \
            --group-add video --cap-add=SYS_PTRACE \
            --security-opt seccomp=unconfined \
            --ipc=host --shm-size 16G \
            -v $MODEL_PATH:/data
            rocm/dev-ubuntu-24.04:7.0-complete

Once inside the Docker container, run the following steps:

3. Set up your workspace:

   .. code-block:: bash

      apt-get update && apt-get install -y nano libcurl4-openssl-dev cmake git
      mkdir -p /workspace && cd /workspace

4. Clone the `https://github.com/ROCm/llama.cpp <https://github.com/ROCm/llama.cpp>`__ repository:

   .. code-block:: bash

      git clone https://github.com/ROCm/llama.cpp
      cd llama.cpp

5. Set your ROCm architecture:

   To compile for supported microarchitectures, run:
   
   .. code-block:: bash

      export LLAMACPP_ROCM_ARCH=gfx942,gfx90a

   .. note::
   
      To compile for a wide range of microarchitectures, run:

      .. code-block:: bash

         export LLAMACPP_ROCM_ARCH=gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102

6. Build and install llama.cpp:

   .. code-block:: bash

      HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
      cmake -S . -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=$LLAMACPP_ROCM_ARCH \
      -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON \
      && cmake --build build --config Release -j$(nproc)

Test the llama.cpp installation
================================================================================

llama.cpp unit tests are optional for validating your installation if you used a
prebuilt Docker image from AMD ROCm Docker Hub.

To run unit tests manually and validate your installation fully, follow these steps:

1. To verify that llama.cpp has been successfully installed, run the Docker container as described in :ref:`build-llama-cpp-docker-image`. 

2. Once inside the container, ensure you have access to the Bash shell.

   .. code-block:: bash
   
      cd /workspace/llama.cpp
      ./build/bin/test-backend-ops

   .. note::

      Running unit tests requires at least one supported AMD GPU.
