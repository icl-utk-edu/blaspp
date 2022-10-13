#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Tests"

# Instead of exiting on the first failed test (bash -e),
# run all the tests and accumulate failures into $err.
err=0

cd test
export OMP_NUM_THREADS=8
./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
(( err += $? ))

./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml
(( err += $? ))

# CUDA or HIP
./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
(( err += $? ))

./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml
(( err += $? ))

print "======================================== Smoke tests"
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install/lib64/blaspp" ..
fi

make
./example_gemm
(( err += $? ))

./example_util
(( err += $? ))

print "======================================== Finished test"
exit ${err}
