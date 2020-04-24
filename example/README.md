BLAS++ Example
================================================================================

This is designed as a minimal, standalone example to demonstrate
how to include and link with BLAS++. This assumes that BLAS++ has
been compiled and installed. There are two options:

## Option 1: Makefile

The Makefile must know the compiler used to compile BLAS++,
CXXFLAGS, and LIBS. Set CXX to the compiler, either in your environment
or in the Makefile. For the flags, there are two more options:

a. Using pkg-config to get CXXFLAGS and LIBS for BLAS++ (recommended).
pkg-config must be able to locate the blas++ package. If it is installed
outside the default search path (see `pkg-config --variable pc_path pkg-config`),
it should be added to `$PKG_CONFIG_PATH`. For instance:
   
    export PKG_CONFIG_PATH=/usr/local/blaspp/lib/pkgconfig  # for sh
    setenv PKG_CONFIG_PATH /usr/local/blaspp/lib/pkgconfig  # for csh
    
b. Hard-code CXXFLAGS and LIBS for BLAS++ in the Makefile.

Then, to build `example_gemm` using the Makefile, run:
    
    make
    
## Option 2: CMake

todo: CMake must know where BLAS++ is installed.

Then, to build `example_gemm` using the CMakeLists.txt, run:

    mkdir build
    cd build
    cmake ..
    make
