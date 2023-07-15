#!/bin/bash -x

maker=$1
device=$2

if [ "${maker}" = "cmake" ]; then
    rm -rf build
    mkdir -p build
fi

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Environment"
env

print "======================================== Setup build"
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

rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config prefix=${top}/install \
         || exit 10
fi
if [ "${maker}" = "cmake" ]; then
    cmake -Dcolor=no \
          -DCMAKE_INSTALL_PREFIX=${top}/install \
          -Dblas_int=${blas_int} \
          -Dgpu_backend=${gpu_backend} .. \
          || exit 12
fi

print "======================================== Finished configure"
exit 0
