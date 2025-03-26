#!/bin/bash -x

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
    make config prefix=${top}/install
    err=$?
    if [[ $err -ne 0 ]]; then
        echo "<<<<<<<<<<<<<<<<<<<< begin config/log.txt"
        cat config/log.txt
        echo ">>>>>>>>>>>>>>>>>>>> end config/log.txt"
        exit 10
    fi

elif [ "${maker}" = "cmake" ]; then

    if [[ $blas != "" ]]; then
        export cmake_blas="-Dblas=$blas"
    fi
    if [[ $blas_int != "" ]]; then
        export cmake_blas_int="-Dblas_int=$blas_int"
    fi
    if [[ $blas_threaded != "" ]]; then
        export cmake_blas_threaded="-Dblas_threaded=$blas_threaded"
    fi
    if [[ $BLAS_LIBRARIES != "" ]]; then
        export cmake_blas_libraries="-DBLAS_LIBRARIES=$BLAS_LIBRARIES"
    fi
    if [[ $bla_vendor != "" ]]; then
        unset cmake_blas
        unset cmake_blas_int
        unset cmake_blas_threaded
        unset cmake_blas_libraries
        export cmake_bla_vendor="-DBLA_VENDOR=$bla_vendor"
    fi

    cmake -Dcolor=no \
          -DCMAKE_INSTALL_PREFIX=${top}/install \
          "$cmake_blas" "$cmake_blas_int" "$cmake_blas_threaded" \
          "$cmake_blas_libraries" "$cmake_bla_vendor" \
          -Dgpu_backend=${gpu_backend} .. \
          || exit 12
fi

cat include/blas/defines.h

print "======================================== Finished configure"
exit 0
