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
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

