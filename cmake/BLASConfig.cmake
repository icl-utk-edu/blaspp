# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#message( "blas config found: " ${blas_config_found} )
if (blas_config_found STREQUAL "TRUE")
    message( "BLAS configuration already done!" )
    return()
endif()

include( "cmake/util.cmake" )

message( STATUS "Checking for BLAS library options" )

if (${blas_defines} MATCHES "HAVE_BLAS")
    #message( STATUS "Checking for library vendors ..." )

    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/mkl_version.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    if (compile_result AND "${run_output}" MATCHES "MKL_VERSION")
        message( "${blue}  ${run_output}${default_color}" )
        set( lib_defines "HAVE_MKL" CACHE INTERNAL "" )
    else()
        set( lib_defines "" )
    endif()
endif()

if (blas_defines MATCHES "HAVE_BLAS"
    AND lib_defines STREQUAL "")
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/acml_version.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}  ${run_output}${default_color}" )
        set( lib_defines "HAVE_ACML" CACHE INTERNAL "" )
    else()
        set( lib_defines "" CACHE INTERNAL "" )
    endif()
endif()

if (${blas_defines} MATCHES "HAVE_BLAS" AND
   "${lib_defines}" STREQUAL "")
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/essl_version.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    if (compile_result AND "${run_output}" MATCHES "ESSL_VERSION")
        message( "${blue}  ${run_output}${default_color}" )
        set( lib_defines "HAVE_ESSL" CACHE INTERNAL "" )
    else()
        set( lib_defines "" CACHE INTERNAL "" )
    endif()
endif()

if (${blas_defines} MATCHES "HAVE_BLAS" AND
    "${lib_defines}" STREQUAL "")
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/openblas_version.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}  ${run_output}${default_color}" )
        set( lib_defines "HAVE_OPENBLAS" CACHE INTERNAL "" )
    else()
        set( lib_defines "" CACHE INTERNAL "" )
    endif()
endif()

message( STATUS "Checking BLAS complex return type..." )

try_run(
    run_result
    compile_result
        ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${blas_int_defines}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)

#message ('compile result: ' ${compile_result})
#message ('run result: ' ${run_result})
#message ('compile output: ' ${compile_output})
#message ('run output: ' ${run_output})

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  BLAS (zdotc) returns complex (GNU gfortran convention)${default_color}" )
    set( blas_complex_return "" )
else()

    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex_argument.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
        )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}  BLAS (zdotc) returns complex as hidden argument (Intel ifort convention)${default_color}" )
        set( blas_complex_return "BLAS_COMPLEX_RETURN_ARGUMENT" )
    else()
        message( FATAL_ERROR "Error - Cannot detect zdotc return value. Please check the BLAS installation." )
    endif()
endif()

message( STATUS "Checking BLAS float return type..." )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/return_float.cc
    LINK_LIBRARIES
        ${blas_links}
        ${blas_cxx_flags}
    COMPILE_DEFINITIONS
        ${blas_int_defines}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}  BLAS (sdot) returns float as float (standard)${default_color}" )
else()
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/return_float_f2c.cc
        LINK_LIBRARIES
            ${blas_links}
            ${blas_cxx_flags}
        COMPILE_DEFINITIONS
            ${blas_int_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}  BLAS (sdot) returns float as double (f2c convention)${default_color}" )
        set( blas_float_return "HAVE_F2C" )
    endif()
endif()

if (DEBUG)
    message( "lib defines: " ${lib_defines} )
    message( "blas defines: " ${blas_defines} )
    message( "mkl int defines: " ${blas_int_defines} )
    message( "fortran mangling defines: " ${fortran_mangling} )
    message( "blas complex return: " ${blas_complex_return} )
    message( "blas float return: " ${blas_float_return} )
    message( "config_found: " ${config_found} )
endif()

if (config_found STREQUAL "TRUE")
    #set( blas_config_found "TRUE" )
    #message( "FOUND BLAS CONFIG" )
    set( blas_config_found "TRUE" CACHE STRING "Set TRUE if BLAS config is found" )
endif()
