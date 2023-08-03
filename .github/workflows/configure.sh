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
# Show environment variables, excluding functions.
(set -o posix; set)

print "======================================== Modules"
quiet module list -l

print "======================================== Query GPUs"
if   [ "${device}" = "gpu_nvidia" ]; then
    nvidia-smi
elif [ "${device}" = "gpu_amd" ]; then
    rocm-smi
elif [ "${device}" = "gpu_intel" ]; then
    clinfo
    sycl-ls
fi

print "======================================== Setup build"
# Note: set all env variables in setup_env.sh,
# else build.sh and test.sh won't see them.

rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config prefix=${top}/install \
         || exit 10

elif [ "${maker}" = "cmake" ]; then
    cmake -Dcolor=no \
          -DCMAKE_INSTALL_PREFIX=${top}/install \
          -Dblas_int=${blas_int} \
          -Dgpu_backend=${gpu_backend} .. \
          || exit 12
fi

cat include/blas/defines.h

print "======================================== Finished configure"
exit 0
