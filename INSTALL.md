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

Makefile Installation
--------------------------------------------------------------------------------

    make           - configures (if make.inc is missing),
                     then compiles the library and tester.
    make config    - configures BLAS++, creating a make.inc file.
    make lib       - compiles the library (lib/libblaspp.so).
    make tester    - compiles test/tester.
    make docs      - generates documentation in docs/html/index.html
    make install   - installs the library and headers to ${prefix}.
    make uninstall - remove installed library and headers from ${prefix}.
    make clean     - deletes object (*.o) and library (*.a, *.so) files.
    make distclean - also deletes make.inc and dependency files (*.d).
    If static=1, makes .a instead of .so library.


### Details

    make config [options]

Runs the `configure.py` script to detect your compiler and library properties,
then creates a make.inc configuration file. You can also manually edit the
make.inc file. Options are name=value pairs to set variables. The configure.py
script can be invoked directly:

    python configure.py [options]

Running `configure.py -h` will print a help message with the current options.
Variables that affect configure.py include:

    CXX                C++ compiler
    CXXFLAGS           C++ compiler flags
    LDFLAGS            linker flags
    CPATH              compiler include search path
    LIBRARY_PATH       compile time library search path
    LD_LIBRARY_PATH    runtime library search path
    DYLD_LIBRARY_PATH  runtime library search path on macOS
    prefix             where to install:
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

The CMake script enforces an out of source build. The simplest way to accomplish
this is to create a build directory off the BLAS++ root directory:

    cd /my/blaspp/dir
    mkdir build && cd build

### Options

By default BLAS++ is set to install into `/opt/slate/`. If you wish to
change this, CMake needs to be told where to install the BLAS++ library.
You can do this by defining CMAKE_INSTALL_PREFIX variable via the CMake
command line:

    # Assuming the working dir is still /my/blaspp/dir/build
    cmake -DCMAKE_INSTALL_PREFIX=/path/to/my/dir ..
    
By default BLAS++ builds a testing suite located in `blaspp/test`.  To disable,
define `BLASPP_BUILD_TESTS` as `OFF`, as follows:

    # Disable building BLASPP test suite
    cmake -DBLASPP_BUILD_TESTS=OFF ..

If `BLASPP_BUILD_TESTS` is enabled, the build will require the TestSweeper
library to be installed via CMake prior to compilation.  Information and 
installation instructions can be found at https://bitbucket.org/icl/testsweeper.

#### BLAS library options

In order the BLAS++ CMake script will:

 1. If `BLAS_LIBRARIES` is set, try the user defined BLAS library.
 2. If `USE_OPTIMIZED_BLAS` is set, try the CMake default `find_package(BLAS)`.
 3. Query the system to find a usable backend.

To specify a particular BLAS backend library, one can define `BLAS_LIBRARIES`
and the configuration script will attempt to link against the explicitly 
provided library.  If this is unsuccessful, this wil generate a CMake error
and exit configuration.

If `USE_OPTIMIZE_BLAS` is set, then `find_package(BLAS)` will be called and, 
if successful, BLAS++ will link against the CMake provided BLAS library.  If
unsuccessful, the last option will be attempted. 

If no other options are selected, or the second is unsuccessful, the BLAS++
CMake script will query the system for a BLAS library by compiling and 
executing several small executables during its configuration.  In order of
precedence, it will find:

 1. Intel MKL
 2. AMD ACML
 3. IBM ESSL
 4. OpenBLAS
 5. Accelerate
 6. BLAS - generically named BLAS library found in the system path
 
If one wishes to limit those options, one can define`BLAS_LIBRARY` as
one of the following strings:

    "Intel MKL"
    "AMD ACML"
    "Cray LibSci"
    "IBM ESSL"
    "OpenBLAS"
    "Apple Accelerate"
    "generic"

This will limit the search to only that library.

### CMake build
Once CMake generates the required makefiles, BLAS++ can be built
and installed using the following:

    # Assuming the working dir is still /my/blaspp/dir/build
    make
    make install
