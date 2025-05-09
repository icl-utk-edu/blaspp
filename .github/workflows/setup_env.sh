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

# append-path VAR PATH
append-path() {
    var=$1
    path=$2
    if [ "${!var}" != "" ]; then
        eval $var+=:$path
    else
        eval $var+=$path
    fi
}

# prepend-path VAR PATH
prepend-path() {
    var=$1
    path=$2
    if [ "${!var}" != "" ]; then
        eval $var=$path:${!var}
    else
        eval $var=$path
    fi
}

# remove-path VAR PATH
remove-path() {
    var=$1
    path=$2
    val=$(echo ${!var} | perl -pe "s%:$path|$path:?%%")
    eval $var=$val
}


#-------------------------------------------------------------------------------
quiet source /etc/profile

hostname && pwd
export top=$(pwd)

shopt -s expand_aliases

print "maker      = '${maker}'"
print "device     = '${device}'"
print "blas       = '${blas}'"
print "blas_int   = '${blas_int}'"
print "bla_vendor = '${bla_vendor}'"
print "check      = '${check}'"

export CPATH LIBRARY_PATH LD_LIBRARY_PATH

if [[ $blas = "mkl" ]] \
    || [[ $bla_vendor = Intel* ]] \
    || [[ $BLAS_LIBRARIES = *mkl* ]]; then
    # See also Intel compiler below.
    quiet module load intel-oneapi-mkl
    print "MKLROOT=${MKLROOT}"

elif [[ $blas = "openblas" ]] \
    || [[ $bla_vendor = "OpenBLAS" ]] \
    || [[ $BLAS_LIBRARIES = *openblas* ]]; then
    # 2025-04: This is int32 version.
    # The openblas/0.3.27/gcc-11.4.1-y3rjih module has libopenblas64_.so
    quiet module load openblas/0.3.27/gcc-11.4.1-jfkp5p

    quiet append-path CPATH           ${ICL_OPENBLAS_ROOT}/include
    quiet append-path LIBRARY_PATH    ${ICL_OPENBLAS_ROOT}/lib
    quiet append-path LD_LIBRARY_PATH ${ICL_OPENBLAS_ROOT}/lib

elif [[ $blas = "blis" ]] \
    || [[ $bla_vendor = "AOCL" ]] \
    || [[ $bla_vendor = "FLAME" ]] \
    || [[ $BLAS_LIBRARIES = *blis* ]]; then
    quiet module load amd-aocl

    quiet append-path CPATH           ${ICL_AMDBLIS_ROOT}/include/blis
    quiet append-path LIBRARY_PATH    ${ICL_AMDBLIS_ROOT}/lib
    quiet append-path LD_LIBRARY_PATH ${ICL_AMDBLIS_ROOT}/lib

    quiet append-path CPATH           ${ICL_AMDLIBFLAME_ROOT}/include
    quiet append-path LIBRARY_PATH    ${ICL_AMDLIBFLAME_ROOT}/lib
    quiet append-path LD_LIBRARY_PATH ${ICL_AMDLIBFLAME_ROOT}/lib
fi

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

#----------------------------------------------------------------- Compiler
if [[ $device = "gpu_intel" ]] \
    || [[ $bla_vendor = Intel* ]]; then
    # Re: bla_vendor = Intel*, apparently, CMake can't find MKL using g++,
    # only using Intel icpx.
    # This is one reason BLAS++ implements its own BLASFinder.
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
