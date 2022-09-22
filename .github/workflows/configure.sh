#!/bin/bash -xe

maker=$1
device=$2

if [ "${maker}" = "cmake" ]; then
    rm -rf build
    mkdir -p build
fi

mydir=$(dirname $0)
source ${mydir}/setup_env.sh

section "======================================== Verify dependencies"
quiet module list
quiet which g++
quiet g++ --version

echo "MKLROOT=${MKLROOT}"

section "======================================== Environment"
env

section "======================================== Setup build"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

