#!/bin/bash -xe

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Build"
make -j8

print "======================================== Install"
make -j8 install
ls -R ${top}/install

print "======================================== Verify build"
ldd_result=$(ldd test/tester)
echo "${ldd_result}"

# Verify that tester linked with cublas or rocblas as intended.
if [ "${device}" = "gpu_nvidia" ]; then
    echo "${ldd_result}" | grep cublas || exit 1
fi
if [ "${device}" = "gpu_amd" ]; then
    echo "${ldd_result}" | grep rocblas || exit 1
fi
