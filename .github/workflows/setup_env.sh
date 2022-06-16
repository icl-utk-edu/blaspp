#!/bin/bash -e 

source /etc/profile

top=`pwd`

module load gcc@7.3.0
module load intel-mkl

if [ "${gpu}" = "nvidia" ]; then
    echo "======================================== Load CUDA"
    export CUDA_HOME=/usr/local/cuda/
    export PATH=${PATH}:${CUDA_HOME}/bin
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
    which nvcc
    nvcc --version
fi

if [ "${gpu}" = "amd" ]; then
    echo "======================================== Load ROCm"
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    which hipcc
    hipcc --version
fi

echo "======================================== Verify dependencies"
module list

which g++
g++ --version

echo "MKLROOT ${MKLROOT}"

echo "======================================== Environment"
env

