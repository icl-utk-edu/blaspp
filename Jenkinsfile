pipeline {

agent none
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
                    values 'caffeine', 'lips'
                }
            } // axes
            stages {
                stage('Build') {
                    agent { node "${host}.icl.utk.edu" }

                    //----------------------------------------------------------
                    steps {
                        sh '''
                        #!/bin/sh +x
                        hostname && pwd

                        source /home/jenkins/spack_setup
                        sload gcc@6.4.0
                        sload intel-mkl

                        # run CUDA tests on lips
                        if [ "${host}" = "lips" ]; then
                            sload cuda
                        fi

                        # run HIP tests on caffeine
                        if [ "${host}" = "caffeine" ]; then
                            if [ -e /opt/rocm ]; then
                                export PATH=${PATH}:/opt/rocm/bin
                                export CPATH=${CPATH}:/opt/rocm/include
                                export LIBRARY_PATH=${LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
                                export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/rocm/lib:/opt/rocm/lib64
                            fi
                        fi

                        echo "========================================"
                        echo "maker ${maker}"
                        if [ "${maker}" = "make" ]; then
                            export color=no
                            make distclean
                            make config CXXFLAGS="-Werror"
                            export top=..
                        fi
                        if [ "${maker}" = "cmake" ]; then
                            sload cmake
                            rm -rf build
                            mkdir build
                            cd build
                            cmake -Dcolor=no -DCMAKE_CXX_FLAGS="-Werror" ..
                            export top=../..
                        fi

                        echo "========================================"
                        make -j8

                        echo "========================================"
                        ldd test/tester

                        echo "========================================"
                        cd test
                        ./run_tests.py --blas1 --blas2 --blas3 --quick --xml ${top}/report-${maker}.xml
                        ./run_tests.py --batch-blas3           --quick --xml ${top}/report-${maker}-batch.xml

                        # CUDA or HIP
                        ./run_tests.py --blas1-device --blas3-device --quick --xml ${top}/report-${maker}-device.xml
                        ./run_tests.py --batch-blas3-device          --quick --xml ${top}/report-${maker}-batch-device.xml
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
