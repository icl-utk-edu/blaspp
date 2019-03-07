# CMake script for BLAS++ library
# repo: http://bitbucket.org/icl/blaspp
# Tests require Libtest testing library from http://bitbucket.org/icl/libtest

# Options:
#  -DCMAKE_INSTALL_PREFIX=<some_dir>
#    Change installation directory, default: /opt/slate/
#  -Dlibtest_DIR=path/to/libtest
#    Point to testing framework used in tests
#  -DBLASPP_BUILD_TESTS=ON
#    Build BLASPP test suite
#  -DBLAS_LIBRARIES=<blas_libraries>
#    Supply specific BLAS implementation, if not usable, CMake will error

cmake_minimum_required(VERSION 3.2)

project(BLASPP
    LANGUAGES CXX
)

option(BLASPP_BUILD_TESTS "Build BLAS++ testers, ON by default" OFF)

# BLAS options
# todo: Goto, BLIS, FLAME, others?
set(BLAS_LIBRARY "auto" CACHE STRING
    "Choose BLAS & LAPACK library: AMD, Apple, Cray, IBM, Intel, OpenBLAS" )
set_property(CACHE BLAS_LIBRARY PROPERTY STRINGS
    "auto" "AMD ACML" "Apple Accelerate" "Cray LibSci" "IBM ESSL"
    "Intel MKL" "OpenBLAS" )

set(BLAS_LIBRARY_MKL "auto" CACHE STRING
    "For Intel MKL: use Intel ifort or GNU gfortran conventions?" )
set_property(CACHE BLAS_LIBRARY_MKL PROPERTY STRINGS
    "auto" "GNU gfortran conventions" "Intel ifort conventions" )

set(BLAS_LIBRARY_INTEGER "auto" CACHE STRING
    "BLAS integer size: int (LP64) or int64_t (ILP64)" )
set_property(CACHE BLAS_LIBRARY_INTEGER PROPERTY STRINGS
    "auto" "int (LP64)" "int64_t (ILP64)" )

set(BLAS_LIBRARY_THREADING "auto" CACHE STRING
    "BLAS threaded or sequential" )
set_property(CACHE BLAS_LIBRARY_THREADING PROPERTY STRINGS
    "auto" "threaded" "sequential" )

# Enforce out of source builds
string (TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" SOURCE_DIR_LOWER)
string (TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" BINARY_DIR_LOWER )
if (SOURCE_DIR_LOWER STREQUAL BINARY_DIR_LOWER)
    message (FATAL_ERROR "Compiling BLASPP with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make")
endif ()

add_library(blaspp SHARED
    src/asum.cc
    src/axpy.cc
    src/batch_gemm.cc
    src/batch_hemm.cc
    src/batch_her2k.cc
    src/batch_herk.cc
    src/batch_symm.cc
    src/batch_syr2k.cc
    src/batch_syrk.cc
    src/batch_trmm.cc
    src/batch_trsm.cc
    src/copy.cc
    src/dot.cc
    src/gemm.cc
    src/gemv.cc
    src/ger.cc
    src/geru.cc
    src/hemm.cc
    src/hemv.cc
    src/her.cc
    src/her2.cc
    src/her2k.cc
    src/herk.cc
    src/iamax.cc
    src/nrm2.cc
    src/rot.cc
    src/rotg.cc
    src/rotm.cc
    src/rotmg.cc
    src/scal.cc
    src/swap.cc
    src/symm.cc
    src/symv.cc
    src/syr.cc
    src/syr2.cc
    src/syr2k.cc
    src/syrk.cc
    src/trmm.cc
    src/trmv.cc
    src/trsm.cc
    src/trsv.cc
)

# CUDA is not required
find_package(CUDA)

if(CUDA_FOUND)
    message(STATUS "Building accelerated CUDA wrappers.")
    target_sources(blaspp PRIVATE
        src/device_batch_gemm.cc
        src/device_batch_hemm.cc
        src/device_batch_her2k.cc
        src/device_batch_herk.cc
        src/device_batch_symm.cc
        src/device_batch_syr2k.cc
        src/device_batch_syrk.cc
        src/device_batch_trmm.cc
        src/device_batch_trsm.cc
        src/device_blas_wrappers.cc
        src/device_error.cc
        src/device_gemm.cc
        src/device_hemm.cc
        src/device_her2k.cc
        src/device_herk.cc
        src/device_queue.cc
        src/device_symm.cc
        src/device_syr2k.cc
        src/device_syrk.cc
        src/device_trmm.cc
        src/device_trsm.cc
        src/device_utils.cc
    )

    target_compile_definitions(blaspp PUBLIC
        BLASPP_WITH_CUBLAS
    )
    target_link_libraries(blaspp PUBLIC
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
    )
    target_include_directories(blaspp PUBLIC
        ${CUDA_INCLUDE_DIRS}
    )
endif()

# Check user supplied BLAS libraries
if(BLAS_LIBRARIES)
    message("BLAS_LIBRARIES supplied by user: " ${BLAS_LIBRARIES})
    include(CheckFortranFunctionExists)
    set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    CHECK_FORTRAN_FUNCTION_EXISTS("dgemm" BLAS_FOUND)
    unset(CMAKE_REQUIRED_LIBRARIES)
    if(BLAS_FOUND)
        message(STATUS "--> BLAS supplied by user is WORKING, will use ${BLAS_LIBRARIES}.")
        target_link_libraries(blaspp ${BLAS_LIBRARIES})
    else()
        message(ERROR "--> BLAS supplied by user is not WORKING, CANNOT USE ${BLAS_LIBRARIES}.")
        message(ERROR "-->     Will use REFERENCE BLAS (by default)")
        message(ERROR "-->     Or Correct your BLAS_LIBRARIES entry ")
        message(ERROR "-->     Or Consider checking USE_OPTIMIZED_BLAS")
    endif()
# Search for system installed BLAS
elseif(USE_OPTIMIZED_BLAS)
    find_package(BLAS)
    if(BLAS_FOUND)
        message("BLAS_FOUND by CMake: " ${BLAS_LIBRARIES})
        target_link_libraries(blaspp ${BLAS_LIBRARIES})
    endif()
else()
    message("BLAS_LIBRARIES not found, using custom search...")

    # Custom BLAS library locator, if all else fails
    include ("cmake/BLASFinder.cmake")
endif()

# Configure BLAS Libraries, determine return types and suck
include ("cmake/BLASConfig.cmake")
if(blas_config_found STREQUAL "TRUE")
    # set blas_config so it the config script isn't run each time
    set (blas_config_found "TRUE" CACHE STRING "Set TRUE if BLAS config is found")
endif()

# Check for CBLAS
if(NOT CBLAS_DEFINES STREQUAL "")
    include ("cmake/CBLASConfig.cmake")
endif()

# Check for LAPACK
if(NOT LAPACK_DEFINES STREQUAL "")
    include ("cmake/LAPACKConfig.cmake")
endif()

# Some debug
if(BLASPP_QUIET)
    message("BLAS DEFINES: " ${BLAS_DEFINES})
    message("BLAS links: " ${BLAS_links})
    message("BLAS LIBRARIES: " ${BLAS_LIBRARIES})
    message("BLAS_cxx_flags: " ${BLAS_cxx_flags})
    message("BLAS INTS defines: " ${BLAS_INT_DEFINES})
    message("FORTRAN MANGLING defines: " ${FORTRAN_MANGLING_DEFINES})
    message("CBLAS DEFINES: " ${CBLAS_DEFINES})
    message("BLAS RETURN: " ${BLAS_RETURN})
    message("LAPACK DEFINES: " ${LAPACK_DEFINES})
endif()

if(NOT ${BLAS_DEFINES} MATCHES "HAVE_BLAS")
    message(FATAL_ERROR "BLAS not found, cannot configure BLAS++")
endif()

target_link_libraries(blaspp PUBLIC
    #LDFLAGS example = -fopenmp
    #LIBS example    = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
    ${BLAS_LIBRARIES}
    ${BLAS_cxx_flags}
)

target_include_directories(blaspp
    PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(blaspp PUBLIC
    # example: -fopenmp -DFORTRAN_ADD_ -DADD_ -DHAVE_BLAS -DBLAS_COMPLEX_RETURN_ARGUMENT -DHAVE_MKL -DHAVE_CBLAS -DHAVE_LAPACK
    ${FORTRAN_MANGLING_DEFINES}
    ${BLAS_DEFINES}
    ${BLAS_RETURN}
    ${LIB_DEFINES}
    ${CBLAS_DEFINES}
    ${LAPACK_DEFINES}
    ${BLAS_int}
)

set_target_properties(blaspp PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
    )

# Set default installation location
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/opt/slate"
        CACHE PATH
        "Install path prefix, prepended onto install directories."
        FORCE
        )
endif()

install (TARGETS blaspp
    EXPORT blasppTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/blaspp/lib
    INCLUDES DESTINATION ${CMAKE_INSTALL_PREFIX}/blaspp/include
)
install(
    DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    DESTINATION
        ${CMAKE_INSTALL_PREFIX}/blaspp
)

export(EXPORT blasppTargets
    FILE ${CMAKE_INSTALL_PREFIX}/blaspp/blasppTargets.cmake
)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/blasppConfig.cmake.in
    ${CMAKE_INSTALL_PREFIX}/blaspp/blasppConfig.cmake
    COPYONLY
)

if (BLASPP_BUILD_TESTS)
    message(STATUS "Blaspp tests will be built.")
    add_subdirectory(test)
endif ()

# To make this more user friendly, add 'make lib' option.
add_custom_target(lib DEPENDS blaspp)