#!/bin/bash

shopt -s expand_aliases

set +x

source /etc/profile

top=`pwd`

# Suppress echo (-x) output of commands executed with `quiet`.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

print_section() {
    builtin echo "$*"
    date
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias section='{ save_flags="$-"; set +x; } 2> /dev/null; print_section'

module load gcc@7.3.0
module load intel-mkl

if [ "${device}" = "gpu_nvidia" ]; then
    section "======================================== Load CUDA"
    export CUDA_HOME=/usr/local/cuda/
    export PATH=${PATH}:${CUDA_HOME}/bin
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
    which nvcc
    nvcc --version
fi

if [ "${device}" = "gpu_amd" ]; then
    section "======================================== Load ROCm"
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    which hipcc
    hipcc --version
fi

if [ "${maker}" = "cmake" ]; then
    section "======================================== Load cmake"
    module load cmake
    which cmake
    cmake --version
    cd build
fi


section "======================================== Verify dependencies"
module list

which g++
g++ --version

echo "MKLROOT=${MKLROOT}"

section "======================================== Environment"
env

set -x

