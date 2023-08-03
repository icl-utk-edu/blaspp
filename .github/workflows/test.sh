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

args="--quick"
if [ "${device}" = "gpu_intel" ]; then
    # Our Intel GPU supports only single precision.
    args+=" --type s,c"
fi

./run_tests.py ${args} --blas1 --blas2 --blas3
(( err += $? ))

./run_tests.py ${args} --batch-blas3
(( err += $? ))

# CUDA, HIP, or SYCL. These fail gracefully when GPUs are absent.
./run_tests.py ${args} --blas1-device --blas3-device
(( err += $? ))

./run_tests.py ${args} --batch-blas3-device
(( err += $? ))

print "======================================== Smoke tests"
cd ${top}/examples

# Makefile or CMakeLists.txt picks up ${test_args}.
if [ "${device}" = "gpu_intel" ]; then
    # Our Intel GPU supports only single precision.
    export test_args="s c"
else
    export test_args="s d c z"
fi

if [ "${maker}" = "make" ]; then
    export PKG_CONFIG_PATH+=:${top}/install/lib/pkgconfig
    make clean || exit 20

elif [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake "-DCMAKE_PREFIX_PATH=${top}/install" .. || exit 30
fi

# ARGS=-V causes CTest to print output. Makefile doesn't use it.
make -j8 || exit 40
make test ARGS=-V
(( err += $? ))

print "======================================== Finished test"
exit ${err}
