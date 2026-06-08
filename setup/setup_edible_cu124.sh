#!/usr/bin/env bash
set -euo pipefail

CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.4}
export PATH
export LD_LIBRARY_PATH
export CUDA_HOME
export CUDACXX=${CUDA_HOME}/bin/nvcc # nvcc requires this (poinnet install)
export PATH=${CUDA_HOME}/bin:${PATH} # nvcc requires this (poinnet install)
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH} # nvcc requires this (poinnet install)
export CPATH=$CPATH:${CUDA_HOME}/targets/x86_64-linux/include # pycuda requires this
export LIBRARY_PATH=$LIBRARY_PATH:${CUDA_HOME}/targets/x86_64-linux/lib # pycuda requires this

sudo apt-get install -y libegl1-mesa-dev

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
CUDA_HOME="$CUDA_HOME" pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation

pip install -e third_party/o3b --no-build-isolation
pip install -e .



# for diff3f
# pip install diffusers transformers accelerate
# pip install pytorch3d==0.7.8+pt2.6.0cu124 --extra-index-url https://miropsota.github.io/torch_packages_builder
# pip install git+https://github.com/skoch9/meshplot.git
# pip install pythreejs


pip install pyrender2
pip install xatlas

