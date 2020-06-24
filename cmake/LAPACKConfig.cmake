# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

if (NOT ${lapack_defines} STREQUAL "")
    message( "CBLAS configuration already done!" )
    return()
endif()

include( "cmake/util.cmake" )

set( local_mangling "-D${fortran_mangling}" )
set( local_int "-D${BLAS_INT_DEFINES}" )

if (NOT "${lib_defines}" STREQUAL "")
    set( local_lib_defines "-D${lib_defines}" )
else()
    set( local_lib_defines "" )
endif()

if (NOT "${blas_defines}" STREQUAL "")
    set( local_blas_defines "-D${blas_defines}" )
else()
    set( local_blas_defines "" )
endif()

if (NOT "${BLAS_INT_DEFINES}" STREQUAL "")
    set( local_int "-D${BLAS_INT_DEFINES}" )
else()
    set( local_int "" )
endif()

message( STATUS "Checking for LAPACK POTRF..." )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_lib_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output
)

#message( "compile_output: ${compile_output1}" )

# if it compiled and ran, then LAPACK is available
if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  Found LAPACK${default_color}" )
    set( lapack_defines "HAVE_LAPACK" CACHE INTERNAL "" )
else()
    message( "${red}  LAPACK not found${default_color}" )
    set( lapack_defines "" CACHE INTERNAL "" )
endif()

set( run_result "" )
set( compile_result "" )
set( run_output "" )
return()
message( STATUS "Checking for LAPACKE POTRF..." )

try_run(
    run_result compile_result
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapacke_potrf.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_lib_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output
)

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  Found LAPACKE${default_color}" )
    set( lapacke_defines "HAVE_LAPACKE" )
else()
    message( "${red}  FAIL${default_color} at (${j},${i} )")
    set( lapacke_defines "" )
endif()

set( run_result "" )
set( compile_result "" )
set( run_output "" )

message( STATUS "Checking for XBLAS..." )

try_run(
    run_result compile_result
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_xblas.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_lib_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output
)

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  Found XBLAS${default_color}" )
    set( xblas_defines "HAVE_XBLAS" )
else()
    message( "${red}  XBLAS not found.${default_color}" )
    set( xblas_defines "" )
endif()
set( run_result "" )
set( compile_result "" )
set( run_output "" )

message( STATUS "Checking LAPACK version..." )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_version.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_lib_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output
)

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  Found LAPACK version number.${default_color}" )

    string( REPLACE "=" ";" run_out_list ${run_output} )
    list( GET run_out_list 1 version_number )
    string( REPLACE "../config" ";" version_list ${version_number} )

    list( GET version_list 0 major_ver )
    list( GET version_list 1 minor_ver )
    list( GET version_list 2 rev_ver )

    # For some reason, the version number strings have extra characters, remove.
    string( REGEX REPLACE "[^0-9]" "" minor_ver ${minor_ver} )
    string( LENGTH ${minor_ver} minor_len )
    if (minor_len LESS 2)
        set( minor_ver "0${minor_ver}" )
    endif()

    # Remove extra characters.
    string( REGEX REPLACE "[^0-9]" "" rev_ver ${rev_ver} )
    string( LENGTH ${rev_ver} rev_len )
    if (rev_len LESS 2)
        set( rev_ver "0${rev_ver}" )
    endif()

    set( lapack_ver_define "LAPACK_VERSION=${major_ver}${minor_ver}${rev_ver}" )
else()
    message( "${red}  Failed to determine LAPACK version.${default_color}" )
    set( lapack_ver_define "" )
endif()

set( run_result "" )
set( compile_result "" )
set( run_output "" )

#message( "lapack defines: " ${lapack_defines} )
#message( "lapacke defines: " ${lapacke_defines} )
#message( "xblas defines: " ${xblas_defines} )
#message( "lapack version define: " ${lapack_ver_define} )
