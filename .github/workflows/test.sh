#!/bin/bash -xe 

maker=$1
gpu=$2

source .github/workflows/setup_env.sh

date
cd test
export OMP_NUM_THREADS=8
./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml

# CUDA or HIP
./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml

echo "======================================== Smoke tests"
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
