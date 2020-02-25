pipeline {
agent none
triggers { cron ('H H(0-2) * * *') }
stages {
stage ('Build'){
parallel {
stage ('Build - Caffeine'){
  agent { node ('caffeine.icl.utk.edu')}
  steps {
    sh '''
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load cmake
      spack load gcc@6.4.0
      spack load intel-mkl
      spack load intel-mpi

      rm -rf *

      hg clone http://bitbucket.org/icl/testsweeper
      cd testsweeper
      make config
      sed -i '/CXXFLAGS/s/$/ -DNO_COLOR/' make.inc
      make
      cd ..

      hg clone http://bitbucket.org/icl/blaspp
      cd blaspp
      make config
      make -j4
    '''
  } // steps
} // build - caffeine
stage ('Build - Lips'){
  agent { node ('lips.icl.utk.edu')}
  steps {
    sh '''
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load cmake
      spack load gcc@6.4.0
      spack load cuda
      spack load intel-mkl
      spack load intel-mpi

      rm -rf *

      hg clone http://bitbucket.org/icl/testsweeper
      cd testsweeper
      make config
      sed -i '/CXXFLAGS/s/$/ -DNO_COLOR/' make.inc
      make
      cd ..

      hg clone http://bitbucket.org/icl/blaspp
      cd blaspp
      make config
      make -j4
    '''
  } // steps
} // build - lips
} // parallel
} // stage (build)
stage ('Test') {
parallel {
stage ('Test - Caffeine') {
  agent { node ('caffeine.icl.utk.edu')}
  steps {
    sh '''
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load gcc@6.4.0
      spack load intel-mkl
      spack load intel-mpi

      cd blaspp/test
      ./run_tests.py --blas1 --blas2 --blas3 --small --xml report1.xml
      ./run_tests.py --batch-blas3 --xsmall --xml report2.xml
    '''
    junit 'blaspp/test/*.xml'
  } // steps
} // stage test caffeine
stage ('Test - Lips') {
  agent { node ('lips.icl.utk.edu')}
  steps {
    sh '''
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load gcc@6.4.0
      spack load cuda
      spack load intel-mkl
      spack load intel-mpi

      cd blaspp/test
      ./run_tests.py --blas1 --blas2 --blas3 --small --xml report1.xml
      ./run_tests.py --batch-blas3 --xsmall --xml report2.xml
    '''
    junit 'blaspp/test/*.xml'
  } // steps
} // stage test lips
} // parallel
} // stage (test)
} // stages
} // pipeline