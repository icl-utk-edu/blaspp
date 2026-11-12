# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

include( "cmake/util.cmake" )

# blaspp_cblas_include given by the user or a parent project, if any.
set( cblas_include_given_ "${blaspp_cblas_include}" )

# Check if this file has already been run with these settings (see bottom).
set( run_ true )
if (DEFINED cblas_config_cache
    AND DEFINED cblas_config_defs_cache
    AND DEFINED cblas_config_include_cache
    AND "${cblas_config_cache}" STREQUAL "${BLAS_LIBRARIES}"
    AND "${cblas_config_include_cache}" STREQUAL "${cblas_include_given_}")

    message( DEBUG "CBLAS config already done for '${BLAS_LIBRARIES}'" )
    # Restore defines found in the previous run (see bottom).
    list( APPEND blaspp_defs_ ${cblas_config_defs_cache} )
    set( run_ false )
endif()

#===============================================================================
# Matching endif at bottom.
if (run_)

# Clear results of a previous run. A blaspp_cblas_include given by the user
# is kept; the directory found below is cached separately.
unset( blaspp_cblas_found         CACHE )
unset( blaspp_cblas_libraries     CACHE )
unset( blaspp_cblas_include_found CACHE )

# Defines found here are appended to blaspp_defs_ after this index.
list( LENGTH blaspp_defs_ defs_begin_ )

#----------------------------------------
# Apple puts cblas.h in weird places. If we can't find it,
# use Accelerate/Accelerate.h, but that had issues compiling with g++. <sigh>
if ("${blaspp_defs_}" MATCHES "HAVE_ACCELERATE")
    set( dir_list
        "/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
        "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers"
    )
    foreach (dir IN LISTS dir_list)
        if (EXISTS "${dir}/cblas.h")
            set( blaspp_cblas_include "${dir}" )
            set( blaspp_cblas_include_found "${dir}" CACHE INTERNAL "" )
            list( APPEND blaspp_defs_ "-DBLAS_HAVE_ACCELERATE_CBLAS_H" )
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
            ${blaspp_defs_}
        CMAKE_FLAGS
            "-DINCLUDE_DIRECTORIES=${blaspp_cblas_include}"
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    # For cross-compiling, assume if it links, the run is okay.
    if (CMAKE_CROSSCOMPILING AND compile_result)
        message( DEBUG "cross: cblas" )
        set( run_result "0"  CACHE STRING "" FORCE )
        set( run_output "ok" CACHE STRING "" FORCE )
    endif()
    debug_try_run( "cblas.cc" "${compile_result}" "${compile_output}"
                              "${run_result}" "${run_output}" )

    if (compile_result AND "${run_output}" MATCHES "ok")
        list( APPEND blaspp_defs_ "-DBLAS_HAVE_CBLAS" )
        set( blaspp_cblas_libraries "${lib}" CACHE INTERNAL "" )
        set( blaspp_cblas_found true CACHE INTERNAL "" )
        break()
    endif()
endforeach()

# Save defines found here, to restore when skipped (see top).
set( defs_ "" )
list( LENGTH blaspp_defs_ defs_end_ )
if (defs_end_ GREATER defs_begin_)
    list( SUBLIST blaspp_defs_ ${defs_begin_} -1 defs_ )
endif()
set( cblas_config_defs_cache "${defs_}" CACHE INTERNAL "" )

endif() # run_
#===============================================================================

# Mark as already run (see top).
set( cblas_config_cache "${BLAS_LIBRARIES}" CACHE INTERNAL "" )
set( cblas_config_include_cache "${cblas_include_given_}" CACHE INTERNAL "" )

# The directory found above takes precedence over one given by the user;
# restore it when skipped.
if (blaspp_cblas_include_found)
    set( blaspp_cblas_include "${blaspp_cblas_include_found}" )
endif()

#-------------------------------------------------------------------------------
if (blaspp_cblas_found)
    message( "${blue}   Found CBLAS library ${blaspp_cblas_libraries}${plain}" )
else()
    message( "${red}   CBLAS library not found. Tester cannot be built.${plain}" )
endif()

message( DEBUG "
blaspp_cblas_found     = '${blaspp_cblas_found}'
blaspp_cblas_libraries = '${blaspp_cblas_libraries}'
blaspp_defs_           = '${blaspp_defs_}'
")
