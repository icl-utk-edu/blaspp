pipeline {
    agent none
    triggers { cron ('H H(0-2) * * *') }
    stages {
        //======================================================================
        stage('Parallel Build') {
            parallel {
                //--------------------------------------------------------------
                stage('Build - Caffeine (gcc 6.4, MKL)') {
                    agent { node 'caffeine.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "BLAS++ Building"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load intel-mkl

                        export color=no
                        make config CXXFLAGS="-Werror"
                        make -j4
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "${currentBuild.fullDisplayName} Caffeine build unstable (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "${currentBuild.fullDisplayName} Caffeine build failed (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Caffeine build failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Caffeine)

                //--------------------------------------------------------------
                stage('Build - Lips (gcc 6.4, CUDA, MKL)') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "BLAS++ Building"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl

                        export color=no
                        make config CXXFLAGS="-Werror"
                        make -j4
                        ldd test/tester
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "${currentBuild.fullDisplayName} Lips build unstable (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "${currentBuild.fullDisplayName} Lips build failed (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Lips build failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                    } // post
                } // stage(Build - Lips)
            } // parallel
        } // stage(Parallel Build)

        //======================================================================
        stage('Parallel Test') {
            parallel {
                //--------------------------------------------------------------
                stage('Test - Caffeine') {
                    agent { node 'caffeine.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "BLAS++ Testing"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load intel-mkl

                        cd test
                        ./run_tests.py --blas1 --blas2 --blas3 --small --xml report.xml
                        ./run_tests.py --batch-blas3 --xsmall --xml report-batch.xml
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "${currentBuild.fullDisplayName} Caffeine test unstable (<${env.BUILD_URL}|Open>)"
                        }
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "${currentBuild.fullDisplayName} Caffeine test failed (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Caffeine test failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit 'test/*.xml'
                        }
                    } // post
                } // stage(Test - Caffeine)

                //--------------------------------------------------------------
                stage('Test - Lips') {
                    agent { node 'lips.icl.utk.edu' }
                    steps {
                        sh '''
                        #!/bin/sh +x
                        echo "BLAS++ Testing"
                        hostname && pwd

                        source /home/jmfinney/spack/share/spack/setup-env.sh
                        spack load gcc@6.4.0
                        spack load cuda
                        spack load intel-mkl

                        cd test
                        ./run_tests.py --blas1 --blas2 --blas3 --small --xml report.xml
                        ./run_tests.py --batch-blas3 --xsmall --xml report-batch.xml
                        '''
                    } // steps
                    post {
                        unstable {
                            slackSend channel: '#slate_ci',
                                color: 'warning',
                                message: "${currentBuild.fullDisplayName} Lips test unstable (<${env.BUILD_URL}|Open>)"
                        }
                        // Lips currently has spurious errors; don't email them.
                        failure {
                            slackSend channel: '#slate_ci',
                                color: 'danger',
                                message: "${currentBuild.fullDisplayName} Lips test failed (<${env.BUILD_URL}|Open>)"
                            mail to: 'slate-dev@icl.utk.edu',
                                subject: "${currentBuild.fullDisplayName} Lips test failed",
                                body: "See more at ${env.BUILD_URL}"
                        }
                        always {
                            junit 'test/*.xml'
                        }
                    } // post
                } // stage(Test - Lips)
            } // parallel
        } // stage(Parallel Test)
    } // stages
} // pipeline
