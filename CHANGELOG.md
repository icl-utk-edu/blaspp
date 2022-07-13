2022.05.00
  - Added Level 3 BLAS template implementations
  - Added device copy, scal
  - Added Schur gemm test, batched tile and LAPACK formats
  - Fixed gbmm flops when rectangular
  - Fixed CMake when BLAS_LIBRARIES is empty

2021.04.01
  - Fixed bug in test_trsm_device for row-major

2021.04.00
  - Added HIP/ROCm support
  - Added include/blas/defines.h based on configuration
  - Various bug and CMake fixes

2020.10.02
  - CMake support for including as subdirectory

2020.10.01
  - Fixes: CMake always checks for CBLAS, in case LAPACK++ needs it

2020.10.00
  - Fixes: CMake defines, version, ILP64; remove [cz]symv prototypes
  - Add `make check`

2020.09.00
  - Clean up namespace
  - Makefile and CMake improvements

2020.08.00
  - Initial release. Functionality:
    - Level 1, 2, 3 BLAS for CPU
    - Level 3 BLAS for GPU
    - Level 3 batched BLAS for CPU and GPU
    - cuBLAS GPU implementation
    - Makefile and CMake build options
