# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Check if this file has already been run with these settings.
if (DEFINED cblas_config_cache
    AND "${cblas_config_cache}" STREQUAL "${BLAS_LIBRARIES}")

    message( DEBUG "CBLAS config already done for '${BLAS_LIBRARIES}'" )
    return()
endif()
set( cblas_config_cache "${BLAS_LIBRARIES}" CACHE INTERNAL "" )

include( "cmake/util.cmake" )

#----------------------------------------
# Apple puts cblas.h in weird places. If we can't find it,
# use Accelerate/Accelerate.h, but that had issues compiling with g++. <sigh>
if ("${blaspp_defines}" MATCHES "HAVE_ACCELERATE")
    set( dir_list
        "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers"
    )
    foreach (dir IN LISTS dir_list)
        if (EXISTS "${dir}/cblas.h")
            set( blaspp_cblas_include "${dir}" )
            set( blaspp_cblas_defines "-DHAVE_ACCELERATE_CBLAS_H" )
            break()
        endif()
    endforeach()
endif()

#-------------------------------------------------------------------------------
set( lib_list ";-lcblas" )
message( DEBUG "lib_list ${lib_list}" )

foreach (lib IN LISTS lib_list)
    message( STATUS "Checking for CBLAS library ${lib}" )

    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/cblas.cc"
        LINK_LIBRARIES
            ${lib} ${BLAS_LIBRARIES} ${openmp_lib} # not "..." quoted; screws up OpenMP
        COMPILE_DEFINITIONS
            "${blaspp_defines}" "${blaspp_cblas_defines}"
        CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${blaspp_cblas_include}"
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "cblas.cc" "${compile_result}" "${compile_output}"
                              "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        set( blaspp_cblas_defines "${blaspp_cblas_defines} -DHAVE_CBLAS" )
        set( blaspp_cblas_libraries "${lib}" CACHE INTERNAL "" )
        set( blaspp_cblas_found true CACHE INTERNAL "" )
        break()
    endif()
endforeach()

# To avoid empty -D, need to strip leading whitespace.
# This seems painful in CMake.
string( STRIP "${blaspp_cblas_defines}" blaspp_cblas_defines )
set( blaspp_cblas_defines "${blaspp_cblas_defines}"
     CACHE INTERNAL "Constants defined for CBLAS" )

#-------------------------------------------------------------------------------
if (blaspp_cblas_found)
    message( "${blue}   Found CBLAS library ${blaspp_cblas_libraries}${plain}" )
else()
    message( "${red}   CBLAS library not found. Tester cannot be built.${plain}" )
endif()

message( DEBUG "
blaspp_cblas_found     = '${blaspp_cblas_found}'
blaspp_cblas_libraries = '${blaspp_cblas_libraries}'
blaspp_cblas_defines   = '${blaspp_cblas_defines}'
")
