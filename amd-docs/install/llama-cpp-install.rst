.. meta::
  :description: installing llama.cpp for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, llama.cpp, GGML

.. _llama-cpp-on-rocm-installation:

********************************************************************
llama.cpp on ROCm installation
********************************************************************

System requirements
====================================================================

To use llama.cpp `b5997 <https://github.com/ROCm/llama.cpp/tree/release/b5997>`__, you need the following prerequisites:

- **ROCm version:** `6.4.0 <https://repo.radeon.com/rocm/apt/6.4/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI210

Key ROCm libraries for llama.cpp
--------------------------------------------------------------------------------

llama.cpp functionality on ROCm is determined by its underlying library
dependencies. These ROCm components affect the capabilities, performance, and
feature set available to developers. Ensure you have the required libraries for 
your corresponding ROCm version.

.. list-table::
    :header-rows: 1

    * - ROCm library
      - ROCm 6.4.x version
      - Purpose
      - Usage
    * - `hipBLAS <https://github.com/ROCm/hipBLAS>`__
      - 2.4.0
      - Provides GPU-accelerated Basic Linear Algebra Subprograms (BLAS) for
        matrix and vector operations.
      - Supports operations such as matrix multiplication, matrix-vector
        products, and tensor contractions. Utilized in both dense and batched
        linear algebra operations.
    * - `hipBLASLt <https://github.com/ROCm/hipBLASLt>`__
      - 0.12.0
      - hipBLASLt is an extension of the hipBLAS library, providing additional
        features like epilogues fused into the matrix multiplication kernel or
        use of integer tensor cores.
      - By setting the flag ``ROCBLAS_USE_HIPBLASLT``, you can dispatch hipBLASLt
        kernels where possible.
    * - `rocWMMA <https://github.com/ROCm/rocWMMA>`__
      - 1.7.0
      - Accelerates warp-level matrix-multiply and matrix-accumulate to speed up matrix
        multiplication (GEMM) and accumulation operations with mixed precision
        support.
      - Can be used to enhance the flash attention performance on AMD compute, by enabling
        the flag at compile time.

Install llama.cpp
================================================================================

To install llama.cpp on ROCm, you have the following options:

- :ref:`Use the prebuilt Docker image <use-docker-with-llama-cpp-pre-installed>` **(recommended)**
- :ref:`Build your own Docker image <build-your-llama-cpp-rocm-docker-image>`

.. _use-docker-with-llama-cpp-pre-installed:

Use a prebuilt Docker image with llama.cpp pre-installed
--------------------------------------------------------------------------------------

Docker is the recommended method to set up a llama.cpp environment, and it avoids 
potential installation issues. The tested, prebuilt image includes llama.cpp, ROCm, 
and other dependencies.

.. important::

   To follow these instructions, input your chosen tag into ``<TAG>``. Example: ``llama.cpp-b5997_rocm6.4.0_ubuntu24.04``.

   Tag endings of ``_full``, ``_server``, and ``_light`` serve different purposes for entrypoints as follows:

   - Full: This image includes both the main executable file and the tools to convert ``LLaMA`` models into ``ggml`` and convert into 4-bit quantization.
   - Server: This image only includes the server executable file.
   - Light: This image only includes the main executable file.

   You can download Docker images with specific ROCm, llama.cpp, and operating system versions. 
   See the available tags on `Docker Hub <https://hub.docker.com/r/rocm/llama.cpp/tags>`_ and see :ref:`docker image support <llama-cpp-docker-support>` below.

1. Download your required public `llama.cpp Docker image <https://hub.docker.com/r/rocm/llama.cpp/tags>`__:

   .. code-block:: bash

      docker pull rocm/llama.cpp:<TAG>_full
      docker pull rocm/llama.cpp:<TAG>_server
      docker pull rocm/llama.cpp:<TAG>_light

2. Launch and connect to the container with the respective entrypoints of your image:

   .. code-block:: bash

      export MODEL_PATH='<your_model_path>'

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

Docker image support
--------------------------------------------------------------------------------------

AMD validates and publishes ready-made `llama.cpp <https://hub.docker.com/r/rocm/llama.cpp>`_ images
with ROCm backends on Docker Hub. The following Docker image tags and associated inventories are
validated for `ROCm 6.4.0 <https://repo.radeon.com/rocm/apt/6.4/>`_.

.. tab-set::

   .. tab-item:: Full Docker

      .. tab-set::

         .. tab-item:: Ubuntu 24.04

            Tag
              `rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_full <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b5997_rocm6.4.0_ubuntu24.04_full/images/sha256-f78f6c81ab2f8e957469415fe2370a1334fe969c381d1fe46050c85effaee9d5>`__

            Inventory
              * `ROCm 6.4.0 <https://repo.radeon.com/rocm/apt/6.4/>`_

   .. tab-item:: Server Docker

      .. tab-set::

         .. tab-item:: Ubuntu 24.04

            Tag
              `rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b5997_rocm6.4.0_ubuntu24.04_server/images/sha256-275ad9e18f292c26a00a2de840c37917e98737a88a3520bdc35fd3fc5c9a6a9b>`__

            Inventory
              * `ROCm 6.4.0 <https://repo.radeon.com/rocm/apt/6.4/>`_

   .. tab-item:: Light Docker

      .. tab-set::

         .. tab-item:: Ubuntu 24.04

            Tag
              `rocm/llama.cpp:llama.cpp-b5997_rocm6.4.0_ubuntu24.04_light <https://hub.docker.com/layers/rocm/llama.cpp/llama.cpp-b5997_rocm6.4.0_ubuntu24.04_light/images/sha256-cc324e6faeedf0e400011f07b49d2dc41a16bae257b2b7befa0f4e2e97231320>`__

            Inventory
              * `ROCm 6.4.0 <https://repo.radeon.com/rocm/apt/6.4/>`_

.. _build-your-llama-cpp-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------

If you want to explore llama.cpp capabilities without being limited to the entrypoints
from the prebuilt Docker images, you have the option to build directly from source inside a
ROCm Ubuntu base Docker image.

The prebuilt base Docker image has all dependencies installed, including:

* ROCm
* hipBLAS
* hipBLASLt
* rocWMMA

1. Start your local container from the base ROCm 6.4.0 image:

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
            rocm/dev-ubuntu-24.04:6.4-complete

Once inside the Docker container, run the following steps:

2. Set up your workspace:

   .. code-block:: bash

      apt-get update && apt-get install -y nano libcurl4-openssl-dev cmake git
      mkdir -p /workspace && cd /workspace

3. Clone the `https://github.com/ROCm/llama.cpp <https://github.com/ROCm/llama.cpp>`__ repository:

   .. code-block:: bash

      git clone https://github.com/ROCm/llama.cpp
      cd llama.cpp

4. Set your ROCm architecture:

   To compile for supported microarchitectures, run:
   
   .. code-block:: bash

      export LLAMACPP_ROCM_ARCH=gfx942,gfx90a

   .. note::
   
      To compile for a wide range of microarchitectures, run:

      .. code-block:: bash

         export LLAMACPP_ROCM_ARCH=gfx803,gfx900,gfx906,gfx908,gfx90a,gfx942,gfx1010,gfx1030,gfx1032,gfx1100,gfx1101,gfx1102

5. Build and install llama.cpp:

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

1. To verify that llama.cpp has been successfully installed, run the Docker container as described in :ref:`build-llama-cpp-docker-image-v25-8`. 

2. Once inside the container, ensure you have access to the Bash shell.

   .. code-block:: bash
   
      cd /workspace/llama.cpp
      ./build/bin/test-backend-ops

   .. note::

      Running unit tests requires at least one AMD GPU.
