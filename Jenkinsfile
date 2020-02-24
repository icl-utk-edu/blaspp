pipeline {
agent none
triggers { cron ('H H(0-2) * * *') }
stages {
parallel {
stage ('Build - Caffeine'){
  agent { node ('caffeine.icl.utk.edu')}
  steps {
    sh ```
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load cmake
      spack load gcc@6.4.0
      spack load cuda
      spack load intel-mkl
      spack load intel-mpi
    ```
  }
}
stage ('Build - Lips'){
  agent { node ('lips.icl.utk.edu')}
  steps {
    sh ```
      #!/bin/sh +x
      echo "BLASPP Building..."
      hostname && pwd

      source /home/jmfinney/spack/share/spack/setup-env.sh
      spack load cmake
      spack load gcc@6.4.0
      spack load cuda
      spack load intel-mkl
      spack load intel-mpi
    ```
  }
}
}
}
}