
echo maker=$maker gpu=$gpu
asdf
echo Done
exit

source /etc/profile

export top=`pwd`

echo "======================================== load compiler"
date

module load gcc@7.3.0
module load intel-mkl

echo "======================================== load CUDA or ROCm"
# Load CUDA.
if [ "${gpu}" = "nvidia" ]; then
    which nvcc
    nvcc --version
    # Load CUDA. 
    export CUDA_HOME=/usr/local/cuda/
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Load HIP.
if [ "${gpu}" = "amd" ]; then
    which hipcc
    hipcc --version
    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
fi

echo "======================================== verify dependencies"
# Check what is loaded.
module list

which g++
g++ --version

echo "MKLROOT ${MKLROOT}"

echo "======================================== env"
env

echo "======================================== setup build"
date
echo "maker ${maker}"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    module load cmake
    which cmake
    cmake --version

    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

echo "======================================== build"
date
make -j8

echo "======================================== install"
date
make -j8 install
ls -R ${top}/install

echo "======================================== verify build"
echo "Verify that tester linked with cublas or rocblas as intended."
date
ldd test/tester
if [ "${gpu}" = "nvidia" ]; then
    ldd test/tester | grep cublas || exit 1
fi
if [ "${gpu}" = "amd" ]; then
    ldd test/tester | grep rocblas || exit 1
fi

echo "======================================== tests"
echo "Run tests."
date
cd test
export OMP_NUM_THREADS=8
./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml

# CUDA or HIP
./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml

echo "======================================== smoke tests"
echo "Verify install with smoke tests."
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
