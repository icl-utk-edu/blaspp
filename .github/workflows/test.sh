#!/bin/bash -x

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

# Instead of exiting on the first failed test (bash -e),
# run all the tests and accumulate failures into $err.
err=0

export OMP_NUM_THREADS=8

print "======================================== Tests"
cd test

TESTER="./run_tests.py --quick"
if [ "${device}" = "gpu_intel" ]; then
    TESTER+=" --type s,c"
fi

${TESTER} --blas1 --blas2 --blas3 --xml ${top}/report-${maker}.xml
(( err += $? ))

${TESTER} --batch-blas3           --xml ${top}/report-${maker}-batch.xml
(( err += $? ))

# CUDA or HIP
${TESTER} --blas1-device --blas3-device --xml ${top}/report-${maker}-device.xml
(( err += $? ))

${TESTER} --batch-blas3-device          --xml ${top}/report-${maker}-batch-device.xml
(( err += $? ))

print "======================================== Smoke tests"
cd ${top}/example

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH=${top}/install/lib/pkgconfig
    make clean
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install" ..
fi

if [ "${device}" = "gpu_intel" ]; then
    TESTS="s c"
else
    TESTS="s d c z"
fi

make
./example_gemm $TESTS
(( err += $? ))

./example_util $TESTS
(( err += $? ))

print "======================================== Finished test"
exit ${err}
