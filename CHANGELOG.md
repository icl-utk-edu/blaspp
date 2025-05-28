2025.05.28 (ABI 2.0.0)
  - Added Level 1 BLAS GPU wrappers
  - Added is_complex_v
  - Added support for BLIS and libFLAME (hence AOCL)
  - Removed support for ACML
  - Removed deprecated enum, memcpy routines
  - Moved [cz]symv and [cz]syr to BLAS++ from LAPACK++, since they
    logically fit in BLAS. Requires linking with an LAPACK library
  - Tester prints stats with --repeat
  - Fixed SYCL include path
  - Fixed testers for n = 0 case
  - Fixed template gemm for beta = 0 to not propagate NaN/Inf (strong zero)

2024.10.26 (ABI 1.0.0)
  - Added PAPI SDE that counts flops
  - Use to_blas_int to convert int32 to int64

2024.05.31 (ABI 1.0.0)
  - Added shared library ABI version
  - Updated enum parameters to have `to_string`, `from_string`;
    deprecate `<enum>2str`, `str2<enum>`
  - Removed some deprecated functions. Deprecated MemcpyKind
  - Added PAPI SDE counters (manually enable for now)
  - Support ROCm 5.6.0 (rocBLAS 3.0), trmm with 3 matrices (A, B, C)
  - Fixed bug in her2k with complex alpha and row-major matrix
  - Fixed bug in example_gemm
  - Use `sqrt`, etc. without `std::` to enable argument dependent lookup (ADL)

2023.11.05
  - Fix Queue workspace
  - Update Fortran strlen handling
  - Fix CMake unity build
  - Fix CMake library ordering

2023.08.25
  - Use yyyy.mm.dd version scheme, instead of yyyy.mm.release
  - Added oneAPI support to CMake
  - Fixed int64 support
  - More robust Makefile configure doesn't require CUDA or ROCm to be in
    compiler search paths (CPATH, LIBRARY_PATH, etc.)

2023.06.00
  - Revised Queue class to allow creating Queue from an existing
    CUDA/HIP stream, cuBLAS/rocBLAS handle, or SYCL queue. Also
    allocates streams and workspace on demand, to make Queue creation
    much lighter weight.
  - Improved oneAPI support

2023.01.00
  - Added oneAPI port (currently Makefile only)
      - Added queue argument to `device_malloc, device_free`, etc.;
        deprecated old versions
      - Deprecated `set_device, get_device`
      - Renamed `device_malloc_pinned` to `host_malloc_pinned`
      - Added `device_copy_{matrix,vector}`;
        deprecated `device_{set,get}{matrix,vector}`
  - Added more Level 1 BLAS on GPU device: axpy, dot, nrm2
  - Moved main repo to https://github.com/icl-utk-edu/blaspp/
  - Refactored routines for better maintainability
  - Use python3

2022.07.00
  - Added workspace in queue; used in LAPACK++
  - Set device in memcpy, etc.
  - Updated Schur gemm test with tile layout

2022.05.00
  - Added Level 3 BLAS template implementations
  - Added device copy, scal
  - Added Schur gemm test, batched tile and LAPACK formats
  - Fixed gbmm flops when rectangular
  - Fixed CMake when BLAS_LIBRARIES is empty

2021.04.01
  - Fixed bug in `test_trsm_device` for row-major

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
