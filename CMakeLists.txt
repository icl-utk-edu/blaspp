# CMake script for BLAS++ library
# repo: http://bitbucket.org/icl/blaspp
# Tests require TestSweeper library from
#     http://bitbucket.org/icl/testsweeper

cmake_minimum_required( VERSION 3.8 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_11 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional

project(
    blaspp
    VERSION 2020.06.00
    LANGUAGES CXX
)

#-------------------------------------------------------------------------------
# Options
option( BUILD_SHARED_LIBS "Build shared libraries" true )
option( build_tests "Build test suite" true )
option( color "Use ANSI color output" true )
option( use_cuda "Use CUDA, if available" true )
option( use_openmp "Use OpenMP, if available" true )
option( use_cmake_find_blas "Use CMake's find_package( BLAS ) rather than the search in BLAS++" false )

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set( CMAKE_INSTALL_PREFIX "/opt/slate"
         CACHE PATH
         "Install path prefix, prepended onto install directories."
         FORCE
    )
    message( STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
    # Append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    # This helps find TestSweeper.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
else()
    message( STATUS "Using CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

# Provide menu of options.
set( BLA_VENDOR "" CACHE STRING
     "BLAS Vendor for use in CMake's FindBLAS. If empty, use BLAS++ search. Some obsolete options are omitted here." )
set_property(
    CACHE BLA_VENDOR PROPERTY STRINGS
    "" All Goto OpenBLAS FLAME ATLAS IBMESSL
    Intel10_32 Intel10_64lp Intel10_64lp_seq Intel10_64ilp Intel10_64ilp_seq
    Intel10_64_dyn Apple NAS Arm Arm_mp Arm_ilp64 Arm_ilp64_mp Generic )

#-----------------------------------
# BLAS options
# todo: Goto, BLIS, FLAME, others?
set( blas "auto" CACHE STRING
     "BLAS library to search for" )
set_property(
    CACHE blas PROPERTY STRINGS
    "auto" "Apple Accelerate" "Cray LibSci" "IBM ESSL"
    "Intel MKL" "OpenBLAS" "AMD ACML" "generic" )

set( blas_fortran "auto" CACHE STRING
     "For Intel MKL: use Intel ifort or GNU gfortran conventions?" )
set_property(
    CACHE blas_fortran PROPERTY STRINGS
    "auto" "GNU gfortran conventions" "Intel ifort conventions" )

set( blas_int "auto" CACHE STRING
     "BLAS integer size: int (LP64) or int64_t (ILP64)" )
set_property(
    CACHE blas_int PROPERTY STRINGS
    "auto" "int (LP64)" "int64_t (ILP64)" )

set( blas_threaded "auto" CACHE STRING
     "Multi-threaded BLAS?" )
set_property(
    CACHE blas_threaded PROPERTY STRINGS
    "auto" "true" "false" )

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling BLAS++ with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
# Build library.
add_library(
    blaspp
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
    src/version.cc
)

# CUDA is not required, use if it is available
if (NOT use_cuda)
    message( STATUS "User has requested to NOT use CUDA" )
else()
    include( CheckLanguage )
    check_language( CUDA )
    if (CMAKE_CUDA_COMPILER)
        enable_language( CUDA )

        # todo: check for cuda 10.0+ and force newer CMake version - if possible
        message( STATUS "Building accelerated CUDA wrappers." )
        target_sources(
            blaspp
            PRIVATE
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

        target_compile_definitions( blaspp PUBLIC "-DBLASPP_WITH_CUBLAS" )
        target_link_libraries( blaspp PRIVATE cublas cudart )
    else()
        message( STATUS "No CUDA support in BLAS++" )
    endif()
endif()

# Include directory.
# During build it's {source}/include; after install it's {prefix}/include.
target_include_directories(
    blaspp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

# OpenMP support.
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        set( openmp_lib "OpenMP::OpenMP_CXX" )
        target_link_libraries( blaspp PUBLIC "${openmp_lib}" )
    endif()
endif()

# Get git commit id.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process( COMMAND git rev-parse --short HEAD
                     OUTPUT_VARIABLE blaspp_id )
    string( STRIP "${blaspp_id}" blaspp_id )
    message( STATUS "blaspp_id = ${blaspp_id}" )
    target_compile_definitions(
        blaspp PRIVATE BLASPP_ID="${blaspp_id}" )
endif()

# Use and export -std=c++11; don't allow -std=gnu++11 extensions.
target_compile_features( blaspp PUBLIC cxx_std_11 )
set_target_properties( blaspp PROPERTIES CXX_EXTENSIONS false )

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        blaspp PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

#-------------------------------------------------------------------------------
# Search for BLAS library.
if (BLA_VENDOR OR use_cmake_find_blas)
    message( DEBUG "Using CMake's FindBLAS" )
    find_package( BLAS )
else()
    message( DEBUG "Using BLASFinder" )
    include( "cmake/BLASFinder.cmake" )
endif()

if (NOT BLAS_FOUND)
    message( FATAL_ERROR "BLAS++ requires a BLAS library and none was found." )
endif()

include( "cmake/BLASConfig.cmake" )
# todo: cblas, lapack only needed for tester.
include( "cmake/CBLASConfig.cmake" )
include( "cmake/LAPACKConfig.cmake" )

# BLAS_LIBRARIES could be private, but then if an application directly
# calls blas, cblas, lapack, lapacke, mkl, essl, etc., it would need to
# devine the exact same BLAS_LIBRARIES. For example, the tester calls
# cblas. Instead, make it public.
target_link_libraries( blaspp PUBLIC "${BLAS_LIBRARIES}" )

target_compile_definitions(
    blaspp PRIVATE "${blas_defines}" "${blas_config_defines}" )

#-------------------------------------------------------------------------------
# Install library and add to <package>Targets.cmake
install(
    TARGETS blaspp
    EXPORT blasppTargets
    LIBRARY DESTINATION "lib"  # no default before 3.14
)

# Install header files
install(
    DIRECTORY "include/"    # / copies contents, not directory itself
    DESTINATION "include"
    FILES_MATCHING REGEX "\.(h|hh)$"
)

# Install <package>Targets.cmake
install(
    EXPORT blasppTargets
    DESTINATION "lib/blaspp"
)

# Also export <package>Targets.cmake in build directory
export(
    EXPORT blasppTargets
    FILE "blasppTargets.cmake"
)

# Install <package>Config.cmake and <package>ConfigVersion.cmake,
# to enable find_package( <package> ).
include( CMakePackageConfigHelpers )
configure_package_config_file(
    "blasppConfig.cmake.in"
    "blasppConfig.cmake"
    INSTALL_DESTINATION "lib/blaspp"
)
write_basic_package_version_file(
    "blasppConfigVersion.cmake"
    VERSION "${blaspp_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/blasppConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/blasppConfigVersion.cmake"
    DESTINATION "lib/blaspp"
)

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# Add 'make lib' target.
add_custom_target( lib DEPENDS blaspp )
