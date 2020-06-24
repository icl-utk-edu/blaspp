# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

if (NOT ${cblas_defines} STREQUAL "")
    message( "CBLAS configuration already done!" )
    return()
endif()

include( "cmake/util.cmake" )

message( STATUS "Checking for CBLAS..." )

#message( "blas_links: " ${blas_links} )
#message( "blas_defines: " ${blas_defines} )
#message( "lib_defines: " ${lib_defines} )
#message( "blas_cxx_flags: " ${blas_cxx_flags} )
#message( "blas_int: " ${blas_int_defines} )

if (NOT "${blas_defines}" STREQUAL "")
    set( local_BLAS_DEFINES "-D${blas_defines}" )
else()
    set( local_BLAS_DEFINES "" )
endif()

if (NOT "${lib_defines}" STREQUAL "")
    set( local_LIB_DEFINES "-D${lib_defines}" )
else()
    set( local_LIB_DEFINES "" )
endif()

#message( "local_LIB_DEFINES: " ${local_LIB_DEFINES} )
#message( "local_BLAS_DEFINES: " ${local_BLAS_DEFINES} )

string( FIND "${blas_links}" "framework" is_accelerate )
#message( "is accelerate: ${is_accelerate}" )
if (NOT ${is_accelerate} STREQUAL "-1")
    set( blas_include_dir "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/" )
    set( blas_inc_dir "-I${blas_include_dir}" )
endif()
#message( "blas_inc_dir: ${blas_include_dir}" )

set( run_output "" )
set( compile_OUTPUT1 "" )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/cblas.cc
    LINK_LIBRARIES
        ${blas_cxx_flags}
        ${blas_links}
    COMPILE_DEFINITIONS
        ${blas_inc_dir}
        ${local_BLAS_DEFINES}
        ${local_LIB_DEFINES}
        ${blas_int_defines}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output
)

#message ('compile result: ' ${compile_result})
#message ('run result: ' ${run_result})
#message ('compile output: ' ${compile_output1})
#message ('run output: ' ${run_output})

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  Found CBLAS${default_color}" )
    set( cblas_defines "HAVE_CBLAS" CACHE INTERNAL "" )
else()
    message( "${red}  CBLAS not found.${default_color}" )
    set( cblas_defines "" CACHE INTERNAL "" )
endif()

set( run_output "" )
set( compile_OUTPUT1 "" )

#message( "cblas defines: " ${cblas_defines} )
