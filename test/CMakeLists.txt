# Can't build testers if TestSweeper is not found.
find_package( testsweeper )
if (testsweeper_FOUND)
    message( STATUS "Found TestSweeper; tester will be built." )
else()
    message( WARNING "TestSweeper not found; tester cannot be built." )
    return()
endif()

#-------------------------------------------------------------------------------
add_executable(
    tester
    test.cc
    test_util.cc
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

include( CheckLanguage )
check_language( CUDA )
if (CMAKE_CUDA_COMPILER)
    enable_language( CUDA )

    message( STATUS "Building CUDA wrapper testers." )
    target_sources(
        tester
        PRIVATE
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
else()
    message( STATUS "No CUDA support" )
endif()

# C++11 is inherited from blaspp, but disabling extensions is not.
set_target_properties( tester PROPERTIES CXX_EXTENSIONS false )

target_link_libraries(
    tester
    testsweeper
    blaspp
)

# Copy run_tests script to build directory. Used in Jenkins.
add_custom_command(
    TARGET tester POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)
