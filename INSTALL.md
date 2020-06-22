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

    CXX                 C++ compiler
    CXXFLAGS            C++ compiler flags
    LDFLAGS             linker flags
    CPATH               compiler include search path
    LIBRARY_PATH        compile-time library search path
    LD_LIBRARY_PATH     runtime library search path
    DYLD_LIBRARY_PATH   runtime library search path on macOS


Makefile Installation
--------------------------------------------------------------------------------

Available targets:

    make           - configures (if make.inc is missing),
                     then compiles the library and tester
    make config    - configures BLAS++, creating a make.inc file
    make lib       - compiles the library (lib/libblaspp.so)
    make tester    - compiles test/tester
    make docs      - generates documentation in docs/html/index.html
    make install   - installs the library and headers to ${prefix}
    make uninstall - remove installed library and headers from ${prefix}
    make clean     - deletes object (*.o) and library (*.a, *.so) files
    make distclean - also deletes make.inc and dependency files (*.d)


### Options

    make config [options]

Runs the `configure.py` script to detect your compiler and library properties,
then creates a make.inc configuration file. You can also manually edit the
make.inc file. Options are name=value pairs to set variables. The configure.py
script can be invoked directly:

    python configure.py [options]

Running `configure.py -h` will print a help message with the current options.
In addition to those listed in the Environment variables section above,
options include:

    static={0,1}        build as shared (default) or static library
    prefix              where to install, default /opt/slate.
                        headers go   in ${prefix}/include,
                        library goes in ${prefix}/lib${LIB_SUFFIX}

These can be set in your environment or on the command line, e.g.,

    python configure.py CXX=g++ prefix=/usr/local

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
    cmake [options] ..
    make
    make install


### Options

CMake uses the settings in the Environment variables section above.
Standard CMake options include:

    BUILD_SHARED_LIBS={ON,off}  build as shared (default) or static library
    CMAKE_INSTALL_PREFIX        where to install, default /opt/slate
    BLA_VENDOR
        use CMake's FindBLAS, instead of BLAS++ search. For values, see:
        https://cmake.org/cmake/help/v3.14/module/FindBLAS.html

BLAS++ specific options include (all values case insensitive):

    blas
        BLAS libraries to search for. One or more of:
        auto            search for all libraries (default)
        LibSci          Cray LibSci
        MKL             Intel MKL
        ESSL            IBM ESSL
        OpenBLAS        OpenBLAS
        Accelerate      Apple Accelerate framework
        ACML            AMD ACML (deprecated)
        generic         -lblas

    blas_int
        BLAS integer size to search for. One or more of:
        auto            search for both sizes (default)
        int             32-bit int (LP64 model)
        int64           64-bit int (ILP64 model)

    blas_threaded
        Whether to search for multi-threaded or sequential BLAS. One or more of:
        auto            search for both threaded and sequential BLAS (default)
        on              multi-threaded BLAS
        off             sequential BLAS

    blas_fortran
        Fortran interface to use. Currently applies only to Intel MKL.
        auto            search for both interfaces (default)
        ifort           use Intel ifort interfaces (e.g., libmkl_intel_lp64)
        gfortran        use GNU gfortran interfaces (e.g., libmkl_gf_lp64)

    BLAS_LIBRARIES
        Specify the exact library, overriding the built-in search. E.g.,
        cmake -DBLAS_LIBRARIES='-lopenblas' ..

    color={ON,off}                use ANSI colors in output
    use_openmp={ON,off}           use OpenMP, if available
    build_tests={ON,off}          build test suite (test/tester)
    use_cmake_find_blas={on,OFF}  use CMake's FindBLAS, instead of BLAS++ search

If `build_tests` is enabled, the build will require the TestSweeper
library to be installed via CMake prior to compilation. Information and
installation instructions can be found at https://bitbucket.org/icl/testsweeper.
Tests also require CBLAS and LAPACK.

These options are defined on the command line using `-D`, e.g.,

    # in build directory
    cmake -Dblas=mkl -Dbuild_tests=off -DCMAKE_INSTALL_PREFIX=/usr/local ..

Alternatively, use the `ccmake` text-based interface or the CMake app GUI.

    # in build directory
    ccmake ..
    Type 'c' to configure, then 'g' to generate Makefile

To re-configure CMake, you may need to delete CMake's cache:

    # in build directory
    rm CMakeCache.txt
    # or
    rm -rf *

To debug the build, set `VERBOSE`:

    # in build directory, after running cmake
    make VERBOSE=1
