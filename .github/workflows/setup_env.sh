#!/bin/bash -e 

source /etc/profile

top=`pwd`

# Suppress echo (-x) output of commands executed with `run`. Useful for Spack.
# set +x, set -x are not echo'd.
run() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

section() {
    { set +x; } 2> /dev/null;
    echo $@;
    date
    set -x
}

# Suppress echo (-x) output of `print` commands. https://superuser.com/a/1141026
# aliasing `echo` causes issues with spack_setup, so use `print` instead.
echo_and_restore() {
    builtin echo "$*"
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'


module load gcc@7.3.0
module load intel-mkl

if [ "${gpu}" = "nvidia" ]; then
    section "======================================== Load CUDA"
    export CUDA_HOME=/usr/local/cuda/
    export PATH=${PATH}:${CUDA_HOME}/bin
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
    which nvcc
    nvcc --version
fi

if [ "${gpu}" = "amd" ]; then
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
fi


section "======================================== Verify dependencies"
module list

which g++
g++ --version

echo "MKLROOT ${MKLROOT}"

section "======================================== Environment"
env

