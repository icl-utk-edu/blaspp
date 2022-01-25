pipeline {

agent none
options {
    // Required to clean before build
    skipDefaultCheckout( true )
}
triggers { pollSCM 'H/10 * * * *' }
stages {
    //======================================================================
    stage('Parallel Build') {
        matrix {
            axes {
                axis {
                    name 'maker'
                    values 'make', 'cmake'
                }
                axis {
                    name 'host'
                    values 'gpu_amd', 'gpu_nvidia'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { label "${host}" }

                    //----------------------------------------------------------
                    steps {
                        cleanWs()
                        checkout scm
                        sh '''
#!/bin/sh

set +e  # errors are not fatal (e.g., Spack sometimes has spurious failures)
set -x  # echo commands

date
hostname && pwd
export top=`pwd`

# Suppress echo (-x) output of commands executed with `run`. Useful for Spack.
# set +x, set -x are not echo'd.
run() {
    { set +x; } 2> /dev/null;
    $@;
    set -x
}

# Suppress echo (-x) output of `print` commands. https://superuser.com/a/1141026
# aliasing `echo` causes issues with spack_setup, so use `print` instead.
echo_and_restore() {
    builtin echo "$*"
    case "$save_flags" in
        (*x*)  set -x
    esac
}
alias print='{ save_flags="$-"; set +x; } 2> /dev/null; echo_and_restore'

print "======================================== load compiler"
date
run source /home/jenkins/spack_setup
run sload gcc@7.3.0
run spack compiler find
run sload intel-mkl

print "======================================== load CUDA or ROCm"
# Load CUDA.
if [ "${host}" = "gpu_nvidia" ]; then
    # Load CUDA. LD_LIBRARY_PATH set by Spack.
    run sload cuda@10.2.89
    export CPATH=${CPATH}:${CUDA_HOME}/include
    export LIBRARY_PATH=${LIBRARY_PATH}:${CUDA_HOME}/lib64
fi

# Load HIP.
if [ "${host}" = "gpu_amd" ]; then
    # Load ROCm/HIP.
    export PATH=${PATH}:/opt/rocm/bin
    export CPATH=${CPATH}:/opt/rocm/include
    export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
fi

print "======================================== verify spack"
# Check what is loaded.
run spack find --loaded

which g++
g++ --version

which nvcc
nvcc --version

which hipcc
hipcc --version

print "MKLROOT ${MKLROOT}"

print "======================================== env"
env

print "======================================== setup build"
date
print "maker ${maker}"
export color=no
rm -rf ${top}/install
if [ "${maker}" = "make" ]; then
    make distclean
    make config CXXFLAGS="-Werror" prefix=${top}/install
fi
if [ "${maker}" = "cmake" ]; then
    run sload cmake
    which cmake
    cmake --version

    rm -rf build && mkdir build && cd build
    cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" \
          -DCMAKE_INSTALL_PREFIX=${top}/install ..
fi

print "======================================== build"
date
make -j8

print "======================================== install"
date
make -j8 install
ls -R ${top}/install

print "======================================== verify build"
print "Verify that tester linked with cublas or rocblas as intended."
date
ldd test/tester
if [ "${host}" = "gpu_nvidia" ]; then
    ldd test/tester | grep cublas || exit 1
fi
if [ "${host}" = "gpu_amd" ]; then
    ldd test/tester | grep rocblas || exit 1
fi

print "======================================== tests"
print "Run tests."
date
cd test
export OMP_NUM_THREADS=8
./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml

# CUDA or HIP
./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml

print "======================================== smoke tests"
print "Verify install with smoke tests."
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
'''
                    } // steps

                    //----------------------------------------------------------
                    post {
                        failure {
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} >> ${STAGE_NAME} >> ${maker} ${host} failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit '*.xml'
                        }
                    } // post

                } // stage(Build)
            } // stages
        } // matrix
    } // stage(Parallel Build)
} // stages

} // pipeline
