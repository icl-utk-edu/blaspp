# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Can't build testers if CBLAS, LAPACK, or TestSweeper are not found.
if (NOT blaspp_cblas_found)
    message( WARNING "CBLAS not found; tester cannot be built." )
    return()
endif()

if (NOT LAPACK_FOUND)
    message( WARNING "LAPACK not found; tester cannot be built." )
    return()
endif()

# Search for TestSweeper library, if not already included (e.g., in SLATE).
message( STATUS "${bold}Checking for TestSweeper library${not_bold}" )
if (NOT TARGET testsweeper)
    find_package( testsweeper QUIET )
    if (testsweeper_FOUND)
        message( "   ${blue}Found TestSweeper library: ${testsweeper_DIR}${plain}" )
        message( "" )
    else()
        set( url "https://github.com/icl-utk-edu/testsweeper" )
        set( tag "v2025.05.28" )
        message( "" )
        message( "---------- TestSweeper" )
        message( STATUS "Fetching TestSweeper ${tag} from ${url}" )
        include( FetchContent )
        FetchContent_Declare( testsweeper GIT_REPOSITORY "${url}"
                                          GIT_TAG "${tag}" )
        FetchContent_MakeAvailable( testsweeper )
        message( "---------- TestSweeper done" )
        message( "" )
    endif()
else()
    message( "   ${blue}TestSweeper already included${plain}" )
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
    test_error.cc
    test_gemm.cc
    test_gemv.cc
    test_ger.cc
    test_hemm.cc
    test_hemv.cc
    test_her.cc
    test_her2.cc
    test_her2k.cc
    test_herk.cc
    test_iamax.cc
    test_max.cc
    test_memcpy.cc
    test_memcpy_2d.cc
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
    test_batch_gemm_device.cc
    test_batch_hemm_device.cc
    test_batch_her2k_device.cc
    test_batch_herk_device.cc
    test_schur_gemm.cc
    test_batch_symm_device.cc
    test_batch_syr2k_device.cc
    test_batch_syrk_device.cc
    test_batch_trmm_device.cc
    test_batch_trsm_device.cc
    test_asum_device.cc
    test_axpy_device.cc
    test_dot_device.cc
    test_iamax_device.cc
    test_nrm2_device.cc
    test_rot_device.cc
    test_rotg_device.cc
    test_rotm_device.cc
    test_rotmg_device.cc
    test_scal_device.cc
    test_swap_device.cc
    test_copy_device.cc
    test_gemm_device.cc
    test_hemm_device.cc
    test_her2k_device.cc
    test_herk_device.cc
    test_symm_device.cc
    test_symv_device.cc
    test_syr2k_device.cc
    test_syrk_device.cc
    test_trmm_device.cc
    test_trsm_device.cc
)

# C++11 is inherited from blaspp, but disabling extensions is not.
set_target_properties( ${tester} PROPERTIES CXX_EXTENSIONS false )

target_link_libraries(
    ${tester}
    testsweeper
    blaspp
    ${blaspp_cblas_libraries}
    ${lapack_libraries_}
)

target_include_directories(
    ${tester}
    PRIVATE
        "${blaspp_cblas_include}"
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
            python3 run_tests.py --quick
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
