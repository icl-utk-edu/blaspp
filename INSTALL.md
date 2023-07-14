BLAS++ Installation Notes
================================================================================

[TOC]

Synopsis
--------------------------------------------------------------------------------

Configure and compile the BLAS++ library and its tester,
then install the headers and library.

Option 1: Makefile

    make && make install

Option 2: CMake

    mkdir build && cd build
    cmake ..
    make && make install


Environment variables (Makefile and CMake)
--------------------------------------------------------------------------------

Standard environment variables affect both Makefile (configure.py) and CMake.
These include:

    LD                  Linker; defaults to CXX
    CXX                 C++ compiler
    CXXFLAGS            C++ compiler flags
    LDFLAGS             linker flags
    CPATH               compiler include search path
    LIBRARY_PATH        compile-time library search path
    LD_LIBRARY_PATH     runtime library search path
    DYLD_LIBRARY_PATH   runtime library search path on macOS
    CUDA_PATH           path to CUDA, e.g., /usr/local/cuda
    CUDA_HOME           also recognized for path to CUDA
    ROCM_PATH           path to ROCm, e.g., /opt/rocm


Options (Makefile and CMake)
--------------------------------------------------------------------------------

BLAS++ specific options include (all values are case insensitive):

    blas
        BLAS libraries to search for. One or more of:
        auto            search for all libraries (default)
        LibSci          Cray LibSci
        MKL             Intel MKL
        ESSL            IBM ESSL
        OpenBLAS        OpenBLAS
        Accelerate      Apple Accelerate framework
        ACML            AMD ACML (deprecated)
        generic         generic -lblas

    blas_int
        BLAS integer size to search for. One or more of:
        auto            search for both sizes (default)
        int32           32-bit int (LP64 model)
        int64           64-bit int (ILP64 model)
        When cross-compiling, this cannot be auto-detected, so the user
        must specify int32 or int64.

    blas_threaded
        Whether to search for multi-threaded or sequential BLAS.
        Currently applies to Intel MKL and IBM ESSL. One of:
        auto            search for both threaded and sequential BLAS (default)
        yes             multi-threaded BLAS
        no              sequential BLAS

    blas_fortran
        Fortran interface to use. Currently applies only to Intel MKL.
        One or more of:
        auto            search for both interfaces (default)
        ifort           use Intel ifort interfaces (e.g., libmkl_intel_lp64)
        gfortran        use GNU gfortran interfaces (e.g., libmkl_gf_lp64)

    fortran_mangling
        (Makefile only; CMake always searches all manglings)
        BLAS and LAPACK are written in Fortran, which has a
        compiler-specific name mangling scheme: routine DGEMM is called
        dgemm_, dgemm, or DGEMM in the library. One or more of:
        auto            search all manglings (default)
        add_            add _ to names  (dgemm_)
        lower           lowercase names (dgemm)
        upper           uppercase names (DGEMM)

    BLAS_LIBRARIES
        Specify the exact BLAS libraries, overriding the built-in search. E.g.,
        cmake -DBLAS_LIBRARIES='-lopenblas' ..

    gpu_backend
        auto            (default) auto-detect CUDA, HIP/ROCm, or SYCL
        cuda            build with CUDA support
        hip             build with HIP/ROCm support
        sycl            build with SYCL and oneMKL support
        none            do not build with GPU backend

    color
        Whether to use ANSI colors in output. One of:
        auto            uses color if output is a TTY
                        (default with Makefile; not support with CMake)
        yes             (default with CMake)
        no

With Makefile, options are specified as environment variables or on the
command line using `option=value` syntax, such as:

    python3 configure.py blas=mkl

With CMake, options are specified on the command line using
`-Doption=value` syntax (not as environment variables), such as:

    cmake -Dblas=mkl ..


Makefile Installation
--------------------------------------------------------------------------------

Available targets:

    make           - configures (if make.inc is missing),
                     then compiles the library and tester
    make config    - configures BLAS++, creating a make.inc file
    make lib       - compiles the library (lib/libblaspp.so)
    make tester    - compiles test/tester
    make check     - run basic checks using tester
    make docs      - generates documentation in docs/html/index.html
    make install   - installs the library and headers to ${prefix}
    make uninstall - remove installed library and headers from ${prefix}
    make clean     - deletes object (*.o) and library (*.a, *.so) files
    make distclean - also deletes make.inc and dependency files (*.d)


### Options

    make config [options]
    or
    python3 configure.py [options]

Runs the `configure.py` script to detect your compiler and library properties,
then creates a make.inc configuration file. You can also manually edit the
make.inc file. Options are name=value pairs to set variables.

Besides the Environment variables and Options listed above, additional
options include:

    static
        Whether to build as a static or shared library.
        0               shared library (default)
        1               static library

    prefix
        Where to install, default /opt/slate.
        Headers go   in ${prefix}/include,
        library goes in ${prefix}/lib${LIB_SUFFIX}

These can be set in your environment or on the command line, e.g.,

    python3 configure.py CXX=g++ prefix=/usr/local

Configure assumes environment variables are set so your compiler can find BLAS
libraries. For example:

    export LD_LIBRARY_PATH="/opt/my-blas/lib64"  # or DYLD_LIBRARY_PATH on macOS
    export LIBRARY_PATH="/opt/my-blas/lib64"
    export CPATH="/opt/my-blas/include"
    or
    export LDFLAGS="-L/opt/my-blas/lib64 -Wl,-rpath,/opt/my-blas/lib64"
    export CXXFLAGS="-I/opt/my-blas/include"

On some systems, loading the appropriate module will set these flags:

    module load my-blas


### Vendor notes

Intel MKL provides scripts to set these flags, e.g.:

    source /opt/intel/bin/compilervars.sh intel64
    or
    source /opt/intel/mkl/bin/mklvars.sh intel64


### Manual configuration

If you have a specific configuration that you want, set CXX, CXXFLAGS, LDFLAGS,
and LIBS, e.g.:

    export CXX="g++"
    export CXXFLAGS="-I${MKLROOT}/include -fopenmp"
    export LDFLAGS="-L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -fopenmp"
    export LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"

These can also be set when running configure:

    make config CXX=g++ \
                CXXFLAGS="-I${MKLROOT}/include -fopenmp" \
                LDFLAGS="-L${MKLROOT}/lib/intel64 -Wl,-rpath,${MKLROOT}/lib/intel64 -fopenmp" \
                LIBS="-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lm"

Note that all test programs are compiled with those options, so errors may cause
configure to fail.

If you experience unexpected problems, please see config/log.txt to diagnose the
issue. The log shows the option being tested, the exact command run, the
command's standard output (stdout), error output (stderr), and exit status. All
test files are in the config directory.


CMake Installation
--------------------------------------------------------------------------------

The CMake script enforces an out-of-source build. Create a build
directory under the BLAS++ root directory:

    cd /path/to/blaspp
    mkdir build && cd build
    cmake [-DCMAKE_INSTALL_PREFIX=/path/to/install] [options] ..
    make
    make install

BLAS++ uses the TestSweeper library (https://github.com/icl-utk-edu/testsweeper)
to run its tests. If CMake doesn't find TestSweeper, it will be
downloaded and compiled. To use a different TestSweeper build that was
not installed, you can point to its directory.

    cmake -Dtestsweeper_DIR=/path/to/testsweeper/build [options] ..


### Options

Besides the Environment variables and Options listed above, additional
options include:

    use_openmp
        Whether to use OpenMP, if available. One of:
        yes (default)
        no

    build_tests
        Whether to build test suite (test/tester).
        Requires TestSweeper, CBLAS, and LAPACK. One of:
        yes (default)
        no

    use_cmake_find_blas
        Whether to use CMake's FindBLAS, instead of BLAS++ search. One of:
        yes
        no (default)
        If BLA_VENDOR is set, it automatically uses CMake's FindBLAS.

    BLA_VENDOR
        Use CMake's FindBLAS, instead of BLAS++ search. For values, see:
        https://cmake.org/cmake/help/latest/module/FindBLAS.html

Standard CMake options include:

    BUILD_SHARED_LIBS
        Whether to build as a static or shared library. One of:
        yes             shared library (default)
        no              static library

    CMAKE_INSTALL_PREFIX (alias prefix)
        Where to install, default /opt/slate.
        Headers go   in ${prefix}/include,
        library goes in ${prefix}/lib

    CMAKE_PREFIX_PATH
        Where to look for CMake packages such as BLAS++ and TestSweeper.

    CMAKE_BUILD_TYPE
        Type of build. One of:
        [empty]         default compiler optimization          (no flags)
        Debug           no optimization, with asserts          (-O0 -g)
        Release         optimized, no asserts, no debug info   (-O3 -DNDEBUG)
        RelWithDebInfo  optimized, no asserts, with debug info (-O2 -DNDEBUG -g)
        MinSizeRel      Release, but optimized for size        (-Os -DNDEBUG)

    CMAKE_MESSAGE_LOG_LEVEL (alias log)
        Level of messages to report. In ascending order:
        FATAL_ERROR, SEND_ERROR, WARNING, AUTHOR_WARNING, DEPRECATION,
        NOTICE, STATUS, VERBOSE, DEBUG, TRACE.
        Particularly, DEBUG or TRACE gives useful information.

With CMake, options are specified on the command line using
`-Doption=value` syntax (not as environment variables), such as:

    # in build directory
    cmake -Dblas=mkl -Dbuild_tests=no -DCMAKE_INSTALL_PREFIX=/usr/local ..

Alternatively, use the `ccmake` text-based interface or the CMake app GUI.

    # in build directory
    ccmake ..
    # Type 'c' to configure, then 'g' to generate Makefile

To re-configure CMake, you may need to delete CMake's cache:

    # in build directory
    rm CMakeCache.txt
    # or
    rm -rf *
    cmake [options] ..

To debug the build, set `VERBOSE`:

    # in build directory, after running cmake
    make VERBOSE=1

### Cross-compiling

For cross-compiling, there are several additional options that the user
must determine and specify manually, since CMake cannot auto-detect
them. See also `blas_int` above.

    blas_complex_return
        How zdotc, etc., returns a complex value. This is required when
        cross-compiling -- there is no default value. One of:
        return          As return value. This is the GNU gfortran convention.
        argument        As a hidden complex output argument. This is the
                        Intel ifort convention.

    blas_return_float_f2c
        Whether sdot, etc., returns float (usual convention) or double
        (f2c convention used in CLAPACK and macOS Accelerate).
        no              returns float. Default except for macOS Accelerate.
        yes             returns double. Default for macOS Accelerate.
