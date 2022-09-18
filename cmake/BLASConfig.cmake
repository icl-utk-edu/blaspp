# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Check if this file has already been run with these settings.
if (DEFINED blas_config_cache
    AND "${blas_config_cache}" STREQUAL "${BLAS_LIBRARIES}")

    message( DEBUG "BLAS config already done for '${BLAS_LIBRARIES}'" )
    return()
endif()
set( blas_config_cache "${BLAS_LIBRARIES}" CACHE INTERNAL "" )

include( "cmake/util.cmake" )

#-------------------------------------------------------------------------------
# Search to identify library and get version. Besides providing the
# specific version, this is also helpful to identify libraries given by
# CMake's find_package( BLAS ) or by the user in BLAS_LIBRARIES.
message( STATUS "Checking BLAS library version" )
set( found false )

#---------------------------------------- Apple Accelerate
# There's no accelerate_version() function that I could find,
# so just look for the framework.
if (NOT found)
    if ("${BLAS_LIBRARIES}" MATCHES "-framework Accelerate|Accelerate.framework")
        message( "${blue}   Accelerate framework${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_ACCELERATE" )
        set( found true )
    endif()
endif()

#---------------------------------------- Intel MKL
if (NOT found)
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/mkl_version.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "mkl_version.cc" "${compile_result}" "${compile_output}"
                                    "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "MKL_VERSION")
        message( "${blue}   ${run_output}${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_MKL" )
        set( found true )
    endif()
endif()

#---------------------------------------- IBM ESSL
if (NOT found)
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/essl_version.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "essl_version.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ESSL_VERSION")
        message( "${blue}   ${run_output}${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_ESSL" )
        set( found true )
    endif()
endif()

#---------------------------------------- OpenBLAS
if (NOT found)
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/openblas_version.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "openblas_version.cc" "${compile_result}" "${compile_output}"
                                         "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "OPENBLAS_VERSION")
        message( "${blue}   ${run_output}${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_OPENBLAS" )
        set( found true )
    endif()
endif()

#---------------------------------------- ACML
if (NOT found)
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/acml_version.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "acml_version.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ACML_VERSION")
        message( "${blue}   ${run_output}${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_ACML" )
        set( found true )
    endif()
endif()

# todo: detect Accelerate
# todo: detect Cray libsci

#-------------------------------------------------------------------------------
message( STATUS "Checking BLAS complex return type" )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex.cc"
    LINK_LIBRARIES
        ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
    COMPILE_DEFINITIONS
        ${blaspp_defs_}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)
debug_try_run( "return_complex.cc" "${compile_result}" "${compile_output}"
                                   "${run_result}" "${run_output}" )

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}   BLAS (zdotc) returns complex (GNU gfortran convention)${plain}" )
    # nothing to define
else()
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex_argument.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "return_complex_argument.cc"
                   "${compile_result}" "${compile_output}"
                   "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}   BLAS (zdotc) returns complex as hidden argument (Intel ifort convention)${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_COMPLEX_RETURN_ARGUMENT" )
    else()
        message( FATAL_ERROR "Error - Cannot detect zdotc return value. Please check the BLAS installation." )
    endif()
endif()

#-------------------------------------------------------------------------------
message( STATUS "Checking BLAS float return type" )

try_run(
    run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/config/return_float.cc"
    LINK_LIBRARIES
        ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
    COMPILE_DEFINITIONS
        ${blaspp_defs_}
    COMPILE_OUTPUT_VARIABLE
        compile_output
    RUN_OUTPUT_VARIABLE
        run_output
)
debug_try_run( "return_float.cc" "${compile_result}" "${compile_output}"
                                 "${run_result}" "${run_output}" )

if (compile_result AND "${run_output}" MATCHES "ok")
    message( "${blue}   BLAS (sdot) returns float as float (standard)${plain}" )
    # nothing to define
else()
    # Detect clapack, macOS Accelerate
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/return_float_f2c.cc"
        LINK_LIBRARIES
            ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            ${blaspp_defs_}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "return_float_f2c.cc" "${compile_result}" "${compile_output}"
                                         "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        message( "${blue}   BLAS (sdot) returns float as double (f2c convention)${plain}" )
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_F2C" )
    endif()
endif()

#-------------------------------------------------------------------------------
message( DEBUG "
blaspp_defs_ = '${blaspp_defs_}'
")
