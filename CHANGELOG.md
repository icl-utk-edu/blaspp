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
