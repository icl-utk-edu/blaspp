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
  }
}
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
  }
}
}
}
}
}