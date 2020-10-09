# Can't build testers if CBLAS, LAPACK, or TestSweeper are not found.
if (NOT cblas_found)
    message( WARNING "CBLAS not found; tester cannot be built." )
    return()
endif()

if (NOT lapack_found)
    message( WARNING "LAPACK not found; tester cannot be built." )
    return()
endif()

message( STATUS "Checking for TestSweeper library" )
find_package( testsweeper QUIET )
if (testsweeper_FOUND)
    message( "   Found TestSweeper library: ${testsweeper_DIR}" )
else()
    set( url "https://bitbucket.org/icl/testsweeper" )
    message( "" )
    message( "---------- TestSweeper" )
    message( STATUS "Fetching TestSweeper from ${url}" )
    include( FetchContent )
    FetchContent_Declare( testsweeper GIT_REPOSITORY "${url}" )
    FetchContent_MakeAvailable( testsweeper )
    message( "---------- TestSweeper done" )
    message( "" )
endif()

#-------------------------------------------------------------------------------
set( tester "${blaspp_}tester" )
add_executable(
    ${tester}
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

if (CMAKE_CUDA_COMPILER)
    message( STATUS "Building CUDA wrapper testers." )
    target_sources(
        ${tester}
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
    message( STATUS "No CUDA support in testers" )
endif()

# C++11 is inherited from blaspp, but disabling extensions is not.
set_target_properties( ${tester} PROPERTIES CXX_EXTENSIONS false )

target_link_libraries(
    ${tester}
    testsweeper
    blaspp
    "${cblas_libraries}"
    "${lapack_libraries}"
)

target_include_directories(
    ${tester}
    PUBLIC
        "${cblas_include}"
)

target_compile_definitions(
    ${tester}
    PRIVATE
        "${blas_defines}"
        "${blas_config_defines}"
        "${cblas_defines}"
        "${lapack_defines}"
)

# Copy run_tests script to build directory.
add_custom_command(
    TARGET ${tester} POST_BUILD
    COMMAND
        cp ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
           ${CMAKE_CURRENT_BINARY_DIR}/run_tests.py
)

if (blaspp_is_project)
    add_custom_target(
        "check"
        COMMAND
            python run_tests.py --quick
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
