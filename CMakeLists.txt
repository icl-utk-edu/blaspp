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
option( use_openmp "Use OpenMP, if available, to clean caches" true )

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set( CMAKE_INSTALL_PREFIX "/opt/slate"
         CACHE PATH
         "Install path prefix, prepended onto install directories."
         FORCE
    )
    message( STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
    # CMake docs claims CMAKE_INSTALL_PREFIX is appended, but that seems
    # to happen before CMAKE_INSTALL_PREFIX is set here, so append it now.
    # This is helpful to find TestSweeper.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
    # Append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

#-----------------------------------
# BLAS options
# todo: Goto, BLIS, FLAME, others?
set( blas "auto" CACHE STRING
     "Choose BLAS & LAPACK library: AMD, Apple, Cray, IBM, Intel, OpenBLAS" )
set_property(
    CACHE blas PROPERTY STRINGS
    "auto" "AMD ACML" "Apple Accelerate" "Cray LibSci" "IBM ESSL"
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
     "BLAS threaded or sequential" )
set_property(
    CACHE blas_threaded PROPERTY STRINGS
    "auto" "threaded" "sequential" )

# Check if user has changed options and they need to be revalidated.
unset( do_library_search )
if (NOT cached_blas MATCHES blas)
    set( do_library_search true )
endif()
if (NOT cached_blas_fortran MATCHES blas_fortran)
    set( do_library_search true )
endif()
if (NOT cached_blas_int MATCHES blas_int)
    set( do_library_search true )
endif()
if (NOT cached_blas_int MATCHES blas_threaded)
    set( do_library_search true )
endif()

set( cached_blas ${blas} CACHE INTERNAL "" )
set( cached_blas_fortran ${blas_fortran} CACHE INTERNAL "" )
set( cached_blas_int ${blas_int} CACHE INTERNAL "" )
set( cached_blas_int ${blas_threaded} CACHE INTERNAL "" )

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir_lower )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir_lower )
if (source_dir_lower STREQUAL binary_dir_lower)
    message( FATAL_ERROR
    "Compiling BLAS++ with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make"
    )
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

set_target_properties(
    blaspp
    PROPERTIES
        CXX_STANDARD 11
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        INTERFACE_COMPILE_FEATURES cxx_std_11
)

# CUDA is not required, use if it is available
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

    target_compile_definitions(
        blaspp
        PUBLIC
            BLASPP_WITH_CUBLAS
    )
    target_link_libraries(
        blaspp
        PUBLIC
            cublas cudart
    )
else()
    message( STATUS "No CUDA support" )
endif()

#message( "use_openmp: ${use_openmp}" )
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        target_link_libraries( blaspp PUBLIC OpenMP::OpenMP_CXX )
    endif()
endif()

if (BLAS_LIBRARIES)
    # Check user supplied BLAS libraries
    message( "Checking BLAS_LIBRARIES supplied by user: " ${BLAS_LIBRARIES} )
    # Check for valid libraries
    include( "cmake/BLASFinder.cmake" )
    if (config_found)
        message( STATUS "--> BLAS supplied by user is WORKING, will use ${BLAS_LIBRARIES}." )
        target_link_libraries( blaspp PUBLIC ${BLAS_LIBRARIES} )
    else()
        message( ERROR "--> BLAS supplied by user is not WORKING, CANNOT USE ${BLAS_LIBRARIES}." )
        message( ERROR "-->     Correct your BLAS_LIBRARIES entry " )
        message( ERROR "-->     Or Consider checking USE_OPTIMIZED_BLAS" )
    endif()
elseif (USE_OPTIMIZED_BLAS)
    # Search for system installed BLAS
    find_package( BLAS )
    if (BLAS_FOUND)
        message( "BLAS_FOUND by CMake: " ${BLAS_LIBRARIES} )
        target_link_libraries( blaspp ${BLAS_LIBRARIES} )
    endif()
else()
    message( "BLAS_LIBRARIES not set, using custom search..." )

    # Use custom BLAS library locator, if all else fails
    if (do_library_search)
        include( "cmake/BLASFinder.cmake" )
    endif()
endif()

if (blas_defines MATCHES "HAVE_BLAS")
    # Configure BLAS Libraries, determine return types and suck
    include( "cmake/BLASConfig.cmake" )
    if (blas_config_found)
        # set blas_config so it the config script isn't run each time
        set( blas_config_found true CACHE STRING "Set TRUE if BLAS config is found" )
    endif()

    # Check for CBLAS
    #message( "check for cblas?" )
    if (NOT cblas_defines STREQUAL "")
        include( "cmake/CBLASConfig.cmake" )
    endif()

    # Check for LAPACK
    #message( "check for lapack?" )
    if (NOT lapack_defines STREQUAL "")
        include( "cmake/LAPACKConfig.cmake" )
    endif()
else()
    message( FATAL_ERROR "BLAS++ requires a BLAS library and none was found." )
endif()

if (blas_cxx_flags)
    string( STRIP ${blas_cxx_flags} blas_cxx_flags )
endif()
if (blas_links)
    string( STRIP ${blas_links} blas_links )
endif()

# Some debug
if (DEBUG)
    message( "blas_defines:     ${blas_defines}" )
    message( "lib_defines:      ${lib_defines}" )
    message( "blas_links:       ${blas_links}" )
    message( "BLAS_LIBRARIES:   ${BLAS_LIBRARIES}" )
    message( "blas_cxx_flags:   ${blas_cxx_flags}" )
    message( "BLAS_INT_DEFINES: ${BLAS_INT_DEFINES}" )
    message( "FORTRAN_MANGLING: ${fortran_mangling}" )
    message( "cblas_defines:    ${cblas_defines}" )
    message( "blas_return:      ${blas_return}" )
    message( "lapack_defines:   ${lapack_defines}" )
endif()

if (NOT blas_defines MATCHES "HAVE_BLAS")
    message( FATAL_ERROR "BLAS not found, cannot configure BLAS++" )
endif()

target_link_libraries(
    blaspp
    PUBLIC
        #LDFLAGS example = -fopenmp
        #LIBS example    = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
        ${blas_links}
        ${blas_cxx_flags}
)

target_include_directories(
    blaspp
    PUBLIC
        #$<INSTALL_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_compile_definitions(
    blaspp
    PUBLIC
        # example: -fopenmp -DFORTRAN_ADD_ -DADD_ -DHAVE_BLAS -DBLAS_COMPLEX_RETURN_ARGUMENT -DHAVE_MKL -DHAVE_CBLAS -DHAVE_LAPACK
        ${fortran_mangling}
        ${blas_defines}
        ${blas_return}
        ${lib_defines}
        #${cblas_defines}
        #${lapack_defines}
        ${blas_int_defines}
)

install(
    TARGETS blaspp
    EXPORT blasppTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

export(
    EXPORT blasppTargets
    FILE blasppTargets.cmake
)

export( PACKAGE blaspp )

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/blasppConfig.cmake.in
    ${CMAKE_CURRENT_SOURCE_DIR}/blasppConfig.cmake
    COPYONLY
)


install(
    EXPORT blasppTargets
    FILE blasppTargets.cmake
    DESTINATION lib/blaspp
)

install(
    DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h*"
)

install(
    FILES ${CMAKE_CURRENT_SOURCE_DIR}/blasppConfig.cmake
    DESTINATION lib/blaspp
)

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# To make this more user friendly, add 'make lib' target.
add_custom_target( lib DEPENDS blaspp )
