# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Check if this file has already been run with these settings.
if ("${lapack_config_cache}" STREQUAL "${BLAS_LIBRARIES}")
    message( DEBUG "LAPACK config already done" )
    return()
endif()
set( lapack_config_cache "${BLAS_LIBRARIES}" CACHE INTERNAL "" )

include( "cmake/util.cmake" )

set( lib_list ";-llapack" )
message( DEBUG "lib_list ${lib_list}" )

foreach (lib IN LISTS lib_list)
    message( STATUS "Checking for LAPACK library ${lib}" )

    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc"
        LINK_LIBRARIES
            "${lib} ${BLAS_LIBRARIES}"
        COMPILE_DEFINITIONS
            "${blas_defines}"
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "lapack_potrf.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        set( lapack_defines "-DHAVE_LAPACK" CACHE INTERNAL "" )
        set( lapack_libraries "${lib}" CACHE INTERNAL "" )
        set( lapack_found true CACHE INTERNAL "" )
        break()
    endif()
endforeach()

#-------------------------------------------------------------------------------
if (BLAS_FOUND)
    message( "${blue}   Found LAPACK library ${lapack_libraries}${plain}" )
else()
    message( "${red}   LAPACK library not found. Testers cannot be built.${plain}" )
endif()

message( DEBUG "
lapack_found        = '${lapack_found}'
lapack_libraries    = '${lapack_libraries}'
lapack_defines      = '${lapack_defines}'")
