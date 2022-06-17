#!/bin/bash -xe 

maker=$1
gpu=$2

mydir=`dirname $0`
source $mydir/setup_env.sh

section "======================================== Build"
make -j8

section "======================================== Install"
make -j8 install
ls -R ${top}/install

section "======================================== Verify build"
# Verify that tester linked with cublas or rocblas as intended.
ldd test/tester
if [ "${gpu}" = "nvidia" ]; then
    ldd test/tester | grep cublas || exit 1
fi
if [ "${gpu}" = "amd" ]; then
    ldd test/tester | grep rocblas || exit 1
fi

