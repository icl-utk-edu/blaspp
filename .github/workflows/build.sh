#!/bin/bash -xe 

maker=$1
gpu=$2

source .github/workflows/setup_env.sh

section "======================================== Setup build"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

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

