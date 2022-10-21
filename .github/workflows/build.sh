#!/bin/bash -xe

maker=$1
device=$2

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

print "======================================== Build"
# make -j8

print "======================================== Install"
# make -j8 install
# ls -R ${top}/install

print "======================================== Verify build"
# ldd_result=$(ldd test/tester)
ldd_result="cublas"
echo "${ldd_result}"

# Verify that tester linked with cublas or rocblas as intended.
if [ "${device}" = "gpu_nvidia" ]; then
    echo "${ldd_result}" | grep cublas || exit 13

elif [ "${device}" = "gpu_amd" ]; then
    echo "${ldd_result}" | grep rocblas || exit 14

else
    echo "${ldd_result}" | grep -P "cublas|rocblas" && exit 15
fi
