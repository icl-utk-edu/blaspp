add_executable(blaspp_test
    test.cc
    test_asum.cc
    test_axpy.cc
    test_batch_gemm.cc
    test_batch_hemm.cc
    test_batch_her2k.cc
    test_batch_herk.cc
    test_batch_symm.cc
    test_batch_syr2k.cc
    test_batch_syrk.cc
    test_batch_trmm.cc
    test_batch_trsm.cc
    test_copy.cc
    test_dot.cc
    test_dotu.cc
    test_error.cc
    test_gemm.cc
    test_gemv.cc
    test_ger.cc
    test_geru.cc
    test_hemm.cc
    test_hemv.cc
    test_her.cc
    test_her2.cc
    test_her2k.cc
    test_herk.cc
    test_iamax.cc
    test_max.cc
    test_nrm2.cc
    test_rot.cc
    test_rotg.cc
    test_rotm.cc
    test_rotmg.cc
    test_scal.cc
    test_swap.cc
    test_symm.cc
    test_symv.cc
    test_syr.cc
    test_syr2.cc
    test_syr2k.cc
    test_syrk.cc
    test_trmm.cc
    test_trmv.cc
    test_trsm.cc
    test_trsv.cc
    cblas_wrappers.cc
    lapack_wrappers.cc
)

if(CUDA_FOUND)
    message(STATUS "Building accelerated CUDA wrappers.")
    target_sources(blaspp_test PRIVATE
        test_batch_gemm_device.cc
        test_batch_hemm_device.cc
        test_batch_her2k_device.cc
        test_batch_herk_device.cc
        test_batch_symm_device.cc
        test_batch_syr2k_device.cc
        test_batch_syrk_device.cc
        test_batch_trmm_device.cc
        test_batch_trsm_device.cc
        test_gemm_device.cc
        test_hemm_device.cc
        test_her2k_device.cc
        test_herk_device.cc
        test_symm_device.cc
        test_syr2k_device.cc
        test_syrk_device.cc
        test_trmm_device.cc
        test_trsm_device.cc
    )
endif()

set_target_properties(blaspp_test PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)

# Set -Dlibtest_DIR=/path/to/libtest
if(libtest_DIR)
    message(STATUS "Found using user supplied libtest_DIR to find libtest")
else()
    find_path(libtest_DIR libtestConfig.cmake
        PATHS
            ${CMAKE_INSTALL_PREFIX}/test
            ${libtest_search_path}
    )
endif()
#message("libtest_DIR: ${libtest_DIR}")

find_package(libtest)

if(NOT BLASPP_BUILD_TESTS)
    set_target_properties(blaspp_test
        EXCLUDE_FROM_ALL
    )
endif()

target_link_libraries(blaspp_test
    libtest
    blaspp
    ${BLAS_links}
)

if(NOT ${is_accelerate} STREQUAL "-1")
    set(blas_include_dir "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/")
else()
    set(blas_include_dir "")
endif()

target_include_directories(blaspp_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        #${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${blas_include_dir}
        #"/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/"
        #${libtest_DIR}/include
)
if(0)
    target_compile_definitions(blaspp_test PRIVATE
        ${FORTRAN_MANGLING_DEFINES}
        ${BLAS_DEFINES}
        ${BLAS_RETURN}
        ${MKL_DEFINES}
        ${CBLAS_DEFINES}
        ${LAPACK_DEFINES}
        ${BLAS_int}
     )
endif()

# Since we can't do 'make test'...
add_custom_target(testing DEPENDS blaspp_test)

# Move test script to build directory
add_custom_command(TARGET blaspp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

#Move test script to build directory
add_custom_command(TARGET blaspp_test POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_BINARY_DIR}/blaspp_test
           ${CMAKE_CURRENT_BINARY_DIR}/test
)
