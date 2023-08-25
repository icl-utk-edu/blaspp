#!/bin/bash

#-------------------------------------------------------------------------------
# Functions

# Suppress echo (-x) output of commands executed with `quiet`.
# Useful for sourcing files, loading modules, spack, etc.
# set +x, set -x are not echo'd.
quiet() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# `print` is like `echo`, but suppresses output of the command itself.
# https://superuser.com/a/1141026
echo_and_restore() {
    builtin echo "$*"
    date
    case "${save_flags}" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=$(pwd)

shopt -s expand_aliases

quiet module load intel-oneapi-mkl
print "MKLROOT=${MKLROOT}"

quiet module load python
quiet which python
quiet which python3
python  --version
python3 --version

quiet module load pkgconf
quiet which pkg-config

# CMake finds CUDA in /usr/local/cuda, so need to explicitly set gpu_backend.
export gpu_backend=none
export color=no
export CXXFLAGS="-Werror -Wno-unused-command-line-argument"

# Test int64 build with make/cuda and cmake/amd.
# Test int32 build with cmake/cuda and make/amd and all others.
if [ "${maker}" = "make" -a "${device}" = "gpu_nvidia" ]; then
    export blas_int=int64
elif [ "${maker}" = "cmake" -a "${device}" = "gpu_amd" ]; then
    export blas_int=int64
else
    export blas_int=int32
fi

#----------------------------------------------------------------- Compiler
if [ "${device}" = "gpu_intel" ]; then
    print "======================================== Load Intel oneAPI compiler"
    quiet module load intel-oneapi-compilers
else
    print "======================================== Load GNU compiler"
    quiet module load gcc@11.3
fi
print "---------------------------------------- Verify compiler"
print "CXX = $CXX"
print "CC  = $CC"
print "FC  = $FC"
${CXX} --version
${CC}  --version
${FC}  --version

#----------------------------------------------------------------- GPU
if [ "${device}" = "gpu_nvidia" ]; then
    print "======================================== Load CUDA"
    quiet module load cuda
    print "CUDA_HOME=${CUDA_HOME}"
    export PATH=${PATH}:${CUDA_HOME}/bin
    export gpu_backend=cuda
    quiet which nvcc
    nvcc --version

elif [ "${device}" = "gpu_amd" ]; then
    print "======================================== Load ROCm"
    export ROCM_PATH=/opt/rocm
    # Some hip utilities require /usr/sbin/lsmod
    export PATH=${PATH}:${ROCM_PATH}/bin:/usr/sbin
    export gpu_backend=hip
    quiet which hipcc
    hipcc --version

elif [ "${device}" = "gpu_intel" ]; then
    # Intel oneAPI SYCL compiler loaded above
    export gpu_backend=sycl
fi

#----------------------------------------------------------------- CMake
if [ "${maker}" = "cmake" ]; then
    print "======================================== Load cmake"
    quiet module load cmake
    quiet which cmake
    cmake --version
    cd build
fi
