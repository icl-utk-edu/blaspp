#!/bin/bash

maker=$1
gpu=$2
echo maker=$maker gpu=$gpu

export top=`pwd`

echo "======================================== load compiler"
date

module load gcc@7.3.0
module load intel-mkl

print "======================================== load CUDA or ROCm"
# Load CUDA.
if [ "${host}" = "gpu_nvidia" ]; then
    # Load CUDA. 
    export CUDA_HOME=/usr/local/cuda/
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Load HIP.
if [ "${host}" = "gpu_amd" ]; then
    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
fi

print "======================================== verify dependencies"
# Check what is loaded.
module list

which g++
g++ --version

which nvcc
nvcc --version

which hipcc
hipcc --version

print "MKLROOT ${MKLROOT}"

print "======================================== env"
env

print "======================================== setup build"
date
print "maker ${maker}"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    run sload cmake
    which cmake
    cmake --version

    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

print "======================================== build"
date
make -j8

print "======================================== install"
date
make -j8 install
ls -R ${top}/install

print "======================================== verify build"
print "Verify that tester linked with cublas or rocblas as intended."
date
ldd test/tester
if [ "${host}" = "gpu_nvidia" ]; then
    ldd test/tester | grep cublas || exit 1
fi
if [ "${host}" = "gpu_amd" ]; then
    ldd test/tester | grep rocblas || exit 1
fi

print "======================================== tests"
print "Run tests."
date
cd test
export OMP_NUM_THREADS=8
./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml

# CUDA or HIP
./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml

print "======================================== smoke tests"
print "Verify install with smoke tests."
date
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake -DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp ..
fi

make
./example_gemm || exit 1
./example_util || exit 1

date
