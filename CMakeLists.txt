# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# CMake script for BLAS++ library.

cmake_minimum_required( VERSION 3.21 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_17 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional
# 3.15 message DEBUG, string REPEAT
# 3.17 find_package( CUDAToolkit )
# 3.21 enable_language( HIP ) support

project(
    blaspp
    VERSION 2025.05.28
    LANGUAGES CXX
)

# See notes in GNUmakefile about using abi-compliance-checker.
# soversion is major ABI version.
set( abi_version 2.0.0 )
string( REPLACE "." ";" abi_list "${abi_version}" )
list( GET abi_list 0 soversion )

include( CheckCXXCompilerFlag )

# When built as a sub-project, add a namespace to make targets unique,
# e.g., `make tester` becomes `make blaspp_tester`.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set( blaspp_is_project true )
    set( blaspp_ "" )
else()
    set( blaspp_is_project false )
    set( blaspp_ "blaspp_" )
endif()

#-------------------------------------------------------------------------------
# Options
if (blaspp_is_project)
    set( log "" CACHE STRING "Shorthand for CMAKE_MESSAGE_LOG_LEVEL" )
    set_property( CACHE log PROPERTY STRINGS
                  FATAL_ERROR SEND_ERROR WARNING AUTHOR_WARNING DEPRECATION
                  NOTICE STATUS VERBOSE DEBUG TRACE )
    if (log)
        set( CMAKE_MESSAGE_LOG_LEVEL "${log}" )
    endif()
endif()

option( BUILD_SHARED_LIBS "Build shared libraries" true )
option( build_tests "Build test suite" "${blaspp_is_project}" )
option( color "Use ANSI color output" true )
option( use_cmake_find_blas "Use CMake's find_package( BLAS ) rather than the search in BLAS++" false )
option( use_openmp "Use OpenMP, if available" true )

set( gpu_backend "auto" CACHE STRING "GPU backend to use" )
set_property( CACHE gpu_backend PROPERTY STRINGS
              auto cuda hip sycl none )

# After color.
include( "cmake/util.cmake" )
include( "cmake/config.cmake" )

# Recognize CTest's BUILD_TESTING flag. (Quotes required.)
if (NOT "${BUILD_TESTING}" STREQUAL "")
    set( build_tests "${BUILD_TESTING}" )
endif()

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND blaspp_is_project)

    set( prefix "/opt/slate" CACHE PATH "Shorthand for CMAKE_INSTALL_PREFIX" )
    set( CMAKE_INSTALL_PREFIX "${prefix}"
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
     "BLAS Vendor for use in CMake's FindBLAS / FindLAPACK. If empty, use BLAS++ search. Some obsolete options are omitted here." )
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
    "auto" "AMD AOCL" "Apple Accelerate" "BLIS" "Cray LibSci" "IBM ESSL"
    "Intel MKL" "OpenBLAS" "generic" )

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

#-----------------------------------
# LAPACK options
# todo: FLAME, others?
set( lapack "auto" CACHE STRING
     "LAPACK library to search for. Often, LAPACK is included in the BLAS library (e.g., -lopenblas contains both)." )
set_property(
    CACHE lapack PROPERTY STRINGS
    "auto" "generic" )

message( DEBUG "Settings:
CMAKE_VERSION          = ${CMAKE_VERSION}
CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}
CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}
BUILD_SHARED_LIBS      = ${BUILD_SHARED_LIBS}
BLA_VENDOR             = ${BLA_VENDOR}
blas                   = ${blas}
blas_fortran           = ${blas_fortran}
blas_int               = ${blas_int}
blas_threaded          = ${blas_threaded}
lapack                 = ${lapack}
build_tests            = ${build_tests}
color                  = ${color}
use_cmake_find_blas    = ${use_cmake_find_blas}
gpu_backend            = ${gpu_backend}
use_openmp             = ${use_openmp}
blaspp_is_project      = ${blaspp_is_project}
blaspp_                = ${blaspp_}
abi_version            = ${abi_version}
soversion              = ${soversion}
" )

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
# Load the CUDA language and packages.
message( "" )
set( blaspp_use_cuda false )  # output in blasppConfig.cmake.in
if (gpu_backend MATCHES "^(auto|cuda)$")
    message( STATUS "${bold}Looking for CUDA${not_bold} (gpu_backend = ${gpu_backend})" )
    if (gpu_backend STREQUAL "cuda")
        find_package( CUDAToolkit REQUIRED )
    else()
        find_package( CUDAToolkit QUIET )
    endif()
    if (CUDAToolkit_FOUND)
        set( gpu_backend "cuda" )
        set( blaspp_defs_cuda_ "-DBLAS_HAVE_CUBLAS" )
        set( blaspp_use_cuda true )

        # Set CUDA architecture _before_ enable_language, otherwise we get some
        # default CMake value seemingly unrelated to the native GPU, e.g., 52.
        #
        # See nvcc --help, in section --gpu-architecture
        # 30  Kepler
        # 50  Maxwell
        # 60  Pascal
        # 70  Volta
        # 75  Turing
        # 80  Ampere
        # 90  Hopper
        # 100 Blackwell ?
        #
        if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
            set( arch "native" )
        else()
            set( arch "70" )  # Volta
        endif()
        set( CMAKE_CUDA_ARCHITECTURES "${arch}" CACHE STRING
            "CUDA architectures, as semi-colon separated list of (mostly) 2-3 digit numbers, e.g., 60;70;80 for Pascal, Volta, Ampere" )
        set_property( CACHE CMAKE_CUDA_ARCHITECTURES PROPERTY STRINGS
                    native all all-major 30 50 60 70 75 80 90 )
        enable_language(CUDA)
        message( STATUS "${blue}Building CUDA support${plain}" )
    else()
        message( STATUS "${red}No CUDA support: CUDA not found${plain}" )
    endif()
else()
    message( STATUS "${red}Skipping CUDA search: gpu_backend = ${gpu_backend}${plain}" )
endif()

#-------------------------------------------------------------------------------
# Load HIP language and packages.
message( "" )
set( blaspp_use_hip false )  # output in blasppConfig.cmake.in
if (NOT CUDAToolkit_FOUND
    AND gpu_backend MATCHES "^(auto|hip)$")

    message( STATUS "${bold}Looking for HIP/ROCm${not_bold} (gpu_backend = ${gpu_backend})" )
    if (gpu_backend STREQUAL "hip")
        find_package( hip REQUIRED )
        find_package( rocblas REQUIRED )
    else()
        find_package( rocblas QUIET )
    endif()
    if (rocblas_FOUND)
        set( gpu_backend "hip" )
        set( blaspp_defs_hip_ "-DBLAS_HAVE_ROCBLAS" )
        set( blaspp_use_hip true )

        # For ROCm >= 3.5, wipe hip-clang specific interface options which are propagated
        set_target_properties( hip::device PROPERTIES INTERFACE_COMPILE_OPTIONS "-fPIC" )
        set_target_properties( hip::device PROPERTIES INTERFACE_LINK_LIBRARIES "hip::host" )

        enable_language( HIP )
        message( STATUS "${blue}Building HIP/ROCm support${plain}" )
    else()
        message( STATUS "${red}No HIP/ROCm support: ROCm not found${plain}" )
    endif()
else()
    message( STATUS "${red}Skipping HIP/ROCm search: gpu_backend = ${gpu_backend}${plain}" )
endif()


#-------------------------------------------------------------------------------
# SYCL support.
message( "" )
set( blaspp_use_sycl false )  # output in blasppConfig.cmake.in
# if cuda or hip were found gpu_backend was set appropriately
if (gpu_backend MATCHES "^(auto|sycl)$")

    message( STATUS "${bold}Looking for oneMKL-SYCL${not_bold} (gpu_backend = ${gpu_backend})" )
    if (TARGET MKL::MKL_DPCPP) # Search for MKL only if not already been found
        set( MKL_FOUND true )
    endif()
    if (NOT MKL_FOUND) # Search for MKL only if not already been found
        if (gpu_backend STREQUAL "sycl")
            find_package( MKL CONFIG REQUIRED QUIET HINTS "$ENV{MKL_ROOT}")
        else()
            find_package( MKL CONFIG QUIET HINTS "$ENV{MKL_ROOT}")
        endif()
    endif()
    # message(STATUS "Available targets: ${MKL_IMPORTED_TARGETS}")

    # Check if compiler supports the SYCL flag
    check_cxx_compiler_flag( "-fsycl" FSYCL_SUPPORT )

    # If oneMKL is found and the compiler supports SYCL then
    # enable oneMKL-SYCL-device support
    if (MKL_FOUND AND FSYCL_SUPPORT)
        set( gpu_backend "sycl" )
        set( blaspp_defs_sycl_ "-DBLAS_HAVE_SYCL;-DBLAS_FORTRAN_ADD_" )
        set( blaspp_use_sycl true )

        # Uncomment to use CMake FindBLAS using BLA_VENDOR
        # if (NOT BLA_VENDOR)
        #    set( BLA_VENDOR Intel10_64lp )
        # endif()

        message( STATUS "${blue}Building oneMKL-SYCL device support${plain}" )
    elseif (gpu_backend STREQUAL "sycl")
        message( FATAL_ERROR "${red}SYCL compiler not found${plain}" )
    else()
        message( STATUS "${red}No oneMKL-SYCL device support: oneMKL or SYCL compiler not found${plain}" )
    endif()
else()
    message( STATUS "${red}Skipping oneMKL-SYCL search: gpu_backend = ${gpu_backend}${plain}" )
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
    src/util.cc
    src/version.cc

    src/device_batch_gemm.cc
    src/device_batch_gemm_group.cc
    src/device_batch_hemm.cc
    src/device_batch_her2k.cc
    src/device_batch_herk.cc
    src/device_batch_symm.cc
    src/device_batch_syr2k.cc
    src/device_batch_syrk.cc
    src/device_batch_trmm.cc
    src/device_batch_trsm.cc
    src/device_error.cc
    src/device_gemm.cc
    src/device_gemv.cc
    src/device_hemm.cc
    src/device_her2k.cc
    src/device_herk.cc
    src/device_iamax.cc
    src/device_queue.cc
    src/device_symm.cc
    src/device_syr2k.cc
    src/device_syrk.cc
    src/device_asum.cc
    src/device_axpy.cc
    src/device_dot.cc
    src/device_nrm2.cc
    src/device_scal.cc
    src/device_swap.cc
    src/device_copy.cc
    src/device_rot.cc
    src/device_rotg.cc
    src/device_rotm.cc
    src/device_rotmg.cc
    src/device_trmm.cc
    src/device_trsm.cc
    src/device_utils.cc
    src/cublas_wrappers.cc
    src/rocblas_wrappers.cc
    src/onemkl_wrappers.cc
)

#-------------------------------------------------------------------------------
# Add CUDA, HIP, or SYCL options.
if (CUDAToolkit_FOUND)
    target_sources(
        blaspp
        PRIVATE
        src/cuda/device_shift_vec.cu
    )
    # Some platforms need these to be public libraries.
    target_link_libraries(
        blaspp PUBLIC CUDA::cudart CUDA::cublas )
elseif (rocblas_FOUND)
    target_sources(
        blaspp
        PRIVATE
        src/hip/device_shift_vec.hip
    )
    # Some platforms need these to be public libraries.
    target_link_libraries(
        blaspp PUBLIC roc::rocblas )
elseif (blaspp_use_sycl)
    target_compile_options( blaspp PUBLIC -fsycl )
    target_link_options( blaspp PUBLIC -fsycl )
    target_link_libraries( blaspp PUBLIC -lmkl_sycl -lsycl -lOpenCL )
endif()

#-------------------------------------------------------------------------------
# Clean stale defines.h from Makefile-based build.
file( REMOVE "${CMAKE_CURRENT_SOURCE_DIR}/include/blas/defines.h" )

# Include directory.
# During build it's {source}/include; after install it's {prefix}/include.
target_include_directories(
    blaspp
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>"  # defines.h
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
        "$<INSTALL_INTERFACE:include>"
)

# OpenMP support.
message( "" )
message( STATUS "${bold}Looking for OpenMP${not_bold}" )
set( openmp_lib "" )
set( blaspp_use_openmp false )  # output in blasppConfig.cmake.in
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        set( openmp_lib "OpenMP::OpenMP_CXX" )
        set( blaspp_use_openmp true )
        target_link_libraries( blaspp PUBLIC "${openmp_lib}" )
    endif()
endif()

# If -latomic is required, add as link library to blaspp.
message( "" )
check_libatomic()
if (libatomic_required)
    target_link_libraries( blaspp PUBLIC "-latomic" )
endif()

# Get git commit id.
message( "" )
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process( COMMAND git rev-parse --short HEAD
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     OUTPUT_VARIABLE blaspp_id )
    string( STRIP "${blaspp_id}" blaspp_id )
    message( STATUS "blaspp_id = ${blaspp_id}" )
    # Don't put in blaspp_defs_ as the quotes cause parsing issues.
    target_compile_definitions(
        blaspp PRIVATE BLASPP_ID="${blaspp_id}" )
endif()

# Use and export -std=c++17.
# CMake inexplicably allows gnu++17 or "decay" to c++11 or 14; prohibit those.
target_compile_features( blaspp PUBLIC cxx_std_17 )
set_target_properties(
    blaspp PROPERTIES
    CXX_STANDARD_REQUIRED true  # prohibit < c++17
    CXX_EXTENSIONS false        # prohibit gnu++17
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    VERSION   "${abi_version}"
    SOVERSION "${soversion}"
)

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        blaspp PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

#-------------------------------------------------------------------------------
# Search for BLAS library.
message( "" )
message( STATUS "${bold}Looking for BLAS libraries and options${not_bold}" )

if (BLA_VENDOR OR use_cmake_find_blas)
    message( STATUS "Using CMake's FindBLAS (BLA_VENDOR=${BLA_VENDOR})" )
    find_package( BLAS )
    if (BLAS_FOUND)
        message( STATUS
                 "Using BLAS++ BLASFinder to check Fortran binding, int size:\n"
                 "   BLAS_LIBRARIES=${BLAS_LIBRARIES}" )
        set( blas "" )
        include( "cmake/BLASFinder.cmake" )
    endif()
else()
    message( STATUS "Using BLAS++ BLASFinder:\n   "
             "blas=${blas}, "
             "blas_int=${blas_int}, "
             "blas_threaded=${blas_threaded}, "
             "blas_fortran=${blas_fortran},\n   "
             "BLAS_LIBRARIES=${BLAS_LIBRARIES}" )
    include( "cmake/BLASFinder.cmake" )
endif()

message( DEBUG "BLAS_FOUND=${BLAS_FOUND}" )
message( DEBUG "BLAS_LIBRARIES=${BLAS_LIBRARIES}" )

if (NOT BLAS_FOUND)
    message( FATAL_ERROR "BLAS++ requires a BLAS library and none was found."
             " Ensure that it is accessible in environment variables"
             " $CPATH, $LIBRARY_PATH, and $LD_LIBRARY_PATH." )
endif()

include( "cmake/BLASConfig.cmake" )

# Only tester needs cblas, but always config it so LAPACK++ tester can use it.
include( "cmake/CBLASConfig.cmake" )

# Export via blasppConfig.cmake
# Needed for finding LAPACK.
set( blaspp_libraries "${BLAS_LIBRARIES};${openmp_lib}" CACHE INTERNAL "" )
message( DEBUG "blaspp_libraries = '${blaspp_libraries}'" )

#-------------------------------------------------------------------------------
# Search for LAPACK library.
message( "" )

if (BLA_VENDOR OR use_cmake_find_blas)
    message( DEBUG "Using CMake's FindLAPACK" )
    find_package( LAPACK )
else()
    message( DEBUG "Using LAPACKFinder" )
    include( "cmake/LAPACKFinder.cmake" )
endif()

if (NOT LAPACK_FOUND)
    message( FATAL_ERROR
             "BLAS++ requires LAPACK for [cz]rot, [cz]syr, [cz]symv." )
endif()

# BLAS++ doesn't need LAPACKConfig.cmake, which checks version, XBLAS, LAPACKE.

#-------------------------------------------------------------------------------
# Cache blaspp_defines that were built in BLASFinder, BLASConfig, etc.
set( blaspp_defines ${blaspp_defs_} ${blaspp_defs_cuda_}
     ${blaspp_defs_hip_} ${blaspp_defs_sycl_}
     CACHE INTERNAL "" )

if (true)
    # Extract definitions as #define VAR or #define VAR VALUE.
    set( blaspp_header_defines "" )
    foreach (def IN LISTS blaspp_defines)
        string( REGEX REPLACE "^-D" "" def "${def}" )
        string( REGEX REPLACE "=" " "  def "${def}" )
        string( APPEND blaspp_header_defines "#define ${def}\n" )
    endforeach()

    # ctime format: Mon Nov 16 15:19:47 2020
    string( TIMESTAMP datetime "%a %b %d %H:%M:%S %Y" )

    # Pass defines via header.
    configure_file(
        include/blas/defines.h.in  # in source dir
        include/blas/defines.h     # in binary dir
    )
else()
    # Pass defines via compiler flags.
    target_compile_definitions(
        blaspp PRIVATE ${blaspp_defines} )
endif()

if (LAPACK_LIBRARIES)
    # Update BLAS libraries with LAPACK libraries.
    set( BLAS_LAPACK_LIBRARIES "${BLAS_LIBRARIES};${LAPACK_LIBRARIES}" )
    # Export via blasppConfig.cmake
    set( blaspp_libraries "${BLAS_LAPACK_LIBRARIES};${openmp_lib}" CACHE INTERNAL "" )
    message( DEBUG "blaspp_libraries = '${blaspp_libraries}'" )
endif()

# blaspp_libraries could be private, but then if an application directly
# calls blas, cblas, lapack, lapacke, mkl, essl, etc., it would need to
# devine the exact same blaspp_libraries. For example, the tester calls
# cblas. Instead, make it public.
target_link_libraries( blaspp PUBLIC ${blaspp_libraries} )

# Add 'make lib' target.
if (blaspp_is_project)
    add_custom_target( lib DEPENDS blaspp )
endif()

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# Install rules.
# GNU Filesystem Conventions
include( GNUInstallDirs )
if (WIN32)
    set( install_configdir "blaspp" )
else()
    set( install_configdir "${CMAKE_INSTALL_LIBDIR}/cmake/blaspp" )
endif()

# Install library and add to <package>Targets.cmake
install(
    TARGETS blaspp
    EXPORT blasppTargets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Install header files
install(
    # / copies contents, not directory itself
    DIRECTORY "${PROJECT_SOURCE_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING REGEX "\\.(h|hh)$"
)
install(
    FILES "${PROJECT_BINARY_DIR}/include/blas/defines.h"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/blas"
)

# Install <package>Targets.cmake
install(
    EXPORT blasppTargets
    DESTINATION "${install_configdir}"
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
    INSTALL_DESTINATION "${install_configdir}"
)
write_basic_package_version_file(
    "blasppConfigVersion.cmake"
    VERSION "${blaspp_VERSION}"
    COMPATIBILITY AnyNewerVersion
)
install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/blasppConfig.cmake"
          "${CMAKE_CURRENT_BINARY_DIR}/blasppConfigVersion.cmake"
    DESTINATION "${install_configdir}"
)
