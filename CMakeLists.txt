cmake_minimum_required(VERSION 3.2)
project(BLASPP CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/modules)

# dependencies
#option(USE_MKL            "Use Intel MKL libraries"               ON)
option(BLASPP_BUILD_TESTS "Build BLAS++ testers, ON by default"   ON)
#option(HAVE_DEVICE        "Enable GPU assisted routines"          ON)

#if (USE_MKL)
#    if (NOT DEFINED ENV{MKLROOT})
#        message(FATAL_ERROR "Environment variable MKL_ROOT is not found")
#    endif()
#endif(USE_MKL)

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

include ("${CMAKE_CURRENT_SOURCE_DIR}/config/BLASConfig.cmake")
include ("${CMAKE_CURRENT_SOURCE_DIR}/config/CBLASConfig.cmake")
include ("${CMAKE_CURRENT_SOURCE_DIR}/config/LAPACKConfig.cmake")

# Some debug
if(BLASPP_QUIET)
    message("BLAS DEFINES: " ${BLAS_DEFINES})
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

if(HAVE_MKL)
    message(STATUS "Using MKL as backend.")
endif(HAVE_MKL)

target_link_libraries(blaspp PUBLIC
    #LDFLAGS example = -fopenmp
    #LIBS example    = -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm
    ${BLAS_links}
    ${BLAS_cxx_flags}
)

target_compile_definitions(blaspp PUBLIC
    #-fopenmp -DFORTRAN_ADD_ -DADD_ -DHAVE_BLAS -DBLAS_COMPLEX_RETURN_ARGUMENT -DHAVE_MKL -DHAVE_CBLAS -DHAVE_LAPACK
    ${FORTRAN_MANGLING_DEFINES}
    ${BLAS_DEFINES}
    ${BLAS_RETURN}
    ${MKL_DEFINES}
    ${CBLAS_DEFINES}
    ${LAPACK_DEFINES}
    ${BLAS_int}
)

target_include_directories(blaspp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(blaspp PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

install (TARGETS blaspp EXPORT blaspp DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)

if (BLASPP_BUILD_TESTS)
    message(STATUS "Blaspp tests will be built.")
    #add_subdirectory(test)
    add_executable(blaspp_test
        test/test.cc
        test/test_asum.cc
        test/test_axpy.cc
        test/test_batch_gemm.cc
        test/test_batch_hemm.cc
        test/test_batch_her2k.cc
        test/test_batch_herk.cc
        test/test_batch_symm.cc
        test/test_batch_syr2k.cc
        test/test_batch_syrk.cc
        test/test_batch_trmm.cc
        test/test_batch_trsm.cc
        test/test_copy.cc
        test/test_dot.cc
        test/test_dotu.cc
        test/test_error.cc
        test/test_gemm.cc
        test/test_gemv.cc
        test/test_ger.cc
        test/test_geru.cc
        test/test_hemm.cc
        test/test_hemv.cc
        test/test_her.cc
        test/test_her2.cc
        test/test_her2k.cc
        test/test_herk.cc
        test/test_iamax.cc
        test/test_max.cc
        test/test_nrm2.cc
        test/test_rot.cc
        test/test_rotg.cc
        test/test_rotm.cc
        test/test_rotmg.cc
        test/test_scal.cc
        test/test_swap.cc
        test/test_symm.cc
        test/test_symv.cc
        test/test_syr.cc
        test/test_syr2.cc
        test/test_syr2k.cc
        test/test_syrk.cc
        test/test_trmm.cc
        test/test_trmv.cc
        test/test_trsm.cc
        test/test_trsv.cc
    )

    if(CUDA_FOUND)
        message(STATUS "Building accelerated CUDA wrappers.")
        target_sources(blaspp_test PRIVATE
            test/test_batch_gemm_device.cc
            test/test_batch_hemm_device.cc
            test/test_batch_her2k_device.cc
            test/test_batch_herk_device.cc
            test/test_batch_symm_device.cc
            test/test_batch_syr2k_device.cc
            test/test_batch_syrk_device.cc
            test/test_batch_trmm_device.cc
            test/test_batch_trsm_device.cc
            test/test_gemm_device.cc
            test/test_hemm_device.cc
            test/test_her2k_device.cc
            test/test_herk_device.cc
            test/test_symm_device.cc
            test/test_syr2k_device.cc
            test/test_syrk_device.cc
            test/test_trmm_device.cc
            test/test_trsm_device.cc
        )
    endif()

    find_package(Libtest)

    target_link_libraries(blaspp_test
        Libtest
        blaspp
    )

    target_include_directories(blaspp_test
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/test
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${Libtest_INCLUDE_DIR}
    )
if(0)
    target_compile_definitions(blaspp_test PUBLIC
        ${FORTRAN_MANGLING_DEFINES}
        ${BLAS_DEFINES}
        ${BLAS_RETURN}
        ${MKL_DEFINES}
        ${CBLAS_DEFINES}
        ${LAPACK_DEFINES}
        ${BLAS_int}
    )
endif()
endif ()


