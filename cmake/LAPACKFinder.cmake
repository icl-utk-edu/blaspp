# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# Convert to list, as lapack_libs is later, to match cached value.
string( REGEX REPLACE "([^ ])( +|\\\;)" "\\1;"    LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" )
string( REGEX REPLACE "-framework;" "-framework " LAPACK_LIBRARIES "${LAPACK_LIBRARIES}" )

message( DEBUG "LAPACK_LIBRARIES '${LAPACK_LIBRARIES}'"        )
message( DEBUG "  cached         '${lapack_libraries_cached}'" )
message( DEBUG "lapack           '${lapack}'"                  )
message( DEBUG "  cached         '${lapack_cached}'"           )
message( DEBUG "" )

#-----------------------------------
# Check if this file has already been run with these settings.
if (LAPACK_LIBRARIES
    AND NOT "${lapack_libraries_cached}" STREQUAL "${LAPACK_LIBRARIES}")
    # Ignore lapack if LAPACK_LIBRARIES changes.
    # Set to empty, rather than unset, so when cmake is invoked again
    # they don't force a search.
    message( DEBUG "clear lapack" )
    set( lapack "" CACHE INTERNAL "" )
elseif (NOT ("${lapack_cached}" STREQUAL "${lapack}"))
    # Ignore LAPACK_LIBRARIES if lapack* changed.
    message( DEBUG "unset LAPACK_LIBRARIES" )
    set( LAPACK_LIBRARIES "" CACHE INTERNAL "" )
else()
    message( DEBUG "LAPACK search already done for
    lapack           = ${lapack}
    LAPACK_LIBRARIES = ${LAPACK_LIBRARIES}" )
    return()
endif()

set( lapack_libraries_cached ${LAPACK_LIBRARIES} CACHE INTERNAL "" )  # updated later
set( lapack_cached           ${lapack}           CACHE INTERNAL "" )

include( "cmake/util.cmake" )

message( STATUS "${bold}Looking for LAPACK libraries and options${not_bold} (lapack = ${lapack})" )

#-------------------------------------------------------------------------------
# Parse options: LAPACK_LIBRARIES, lapack.

#---------------------------------------- LAPACK_LIBRARIES
if (LAPACK_LIBRARIES)
    set( test_lapack_libraries true )
endif()

#---------------------------------------- lapack
string( TOLOWER "${lapack}" lapack_ )

if ("${lapack_}" MATCHES "auto")
    set( test_all true )
endif()

if ("${lapack_}" MATCHES "default")
    set( test_default true )
endif()

if ("${lapack_}" MATCHES "generic")
    set( test_generic true )
endif()

message( DEBUG "
LAPACK_LIBRARIES      = '${LAPACK_LIBRARIES}'
lapack                = '${lapack}'
lapack_               = '${lapack_}'
test_lapack_libraries = '${test_lapack_libraries}'
test_default          = '${test_default}'
test_generic          = '${test_generic}'
test_all              = '${test_all}'")

#-------------------------------------------------------------------------------
# Build list of libraries to check.
# todo: add flame?
# todo: LAPACK_?(ROOT|DIR)

set( lapack_libs_list "" )

#---------------------------------------- LAPACK_LIBRARIES
if (test_lapack_libraries)
    # Escape ; semi-colons so we can append it as one item to a list.
    string( REPLACE ";" "\\;" LAPACK_LIBRARIES_ESC "${LAPACK_LIBRARIES}" )
    message( DEBUG "LAPACK_LIBRARIES ${LAPACK_LIBRARIES}" )
    message( DEBUG "   =>          ${LAPACK_LIBRARIES_ESC}" )
    list( APPEND lapack_libs_list "${LAPACK_LIBRARIES_ESC}" )
endif()

#---------------------------------------- default (in BLAS library)
if (test_all OR test_default)
    list( APPEND lapack_libs_list " " )
endif()

#---------------------------------------- generic -llapack
if (test_all OR test_generic)
    list( APPEND lapack_libs_list "-llapack" )
endif()

message( DEBUG "lapack_libs_list ${lapack_libs_list}" )

#-------------------------------------------------------------------------------
# Check each LAPACK library.
# BLAS++ needs only a limited subset of LAPACK, so check for potrf (Cholesky).
# LAPACK++ checks for pstrf (Cholesky with pivoting) to make sure it is
# a complete LAPACK library, since some BLAS libraries (ESSL, ATLAS)
# contain only an optimized subset of LAPACK routines.

unset( LAPACK_FOUND CACHE )
unset( lapackpp_defs_ CACHE )

foreach (lapack_libs IN LISTS lapack_libs_list)
    if ("${lapack_libs}" MATCHES "^ *$")
        set( label "   In BLAS library" )
    else()
        set( label "   ${lapack_libs}" )
    endif()
    pad_string( "${label}" 50 label )

    # Try to link and run LAPACK routine with the library.
    try_run(
        run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc"
        LINK_LIBRARIES
            # Use blaspp_libraries instead of blaspp, when SLATE includes
            # blaspp and lapackpp, so the blaspp library doesn't exist yet.
            # Not "quoted"; screws up OpenMP.
            ${lapack_libs} ${blaspp_libraries}
        COMPILE_DEFINITIONS
            ${blaspp_defines}
        COMPILE_OUTPUT_VARIABLE
            compile_output
        RUN_OUTPUT_VARIABLE
            run_output
    )
    debug_try_run( "lapack_potrf.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

    if (NOT compile_result)
        message( "${label} ${red} no (didn't link: routine not found)${plain}" )
    elseif ("${run_result}" EQUAL 0 AND "${run_output}" MATCHES "ok")
        # If it runs (exits 0), we're done, so break loop.
        message( "${label} ${blue} yes${plain}" )

        set( LAPACK_FOUND true CACHE INTERNAL "" )
        string( STRIP "${lapack_libs}" lapack_libs )
        set( LAPACK_LIBRARIES "${lapack_libs}" CACHE STRING "" FORCE )
        list( APPEND lapackpp_defs_ "-DLAPACK_HAVE_LAPACK" )
        break()
    else()
        message( "${label} ${red} no (didn't run: int mismatch, etc.)${plain}" )
    endif()
endforeach()

# Update to found LAPACK library.
set( lapack_libraries_cached ${LAPACK_LIBRARIES} CACHE INTERNAL "" )

#-------------------------------------------------------------------------------
if (LAPACK_FOUND)
    if (NOT LAPACK_LIBRARIES)
        message( "${blue}   Found LAPACK library in BLAS library${plain}" )
    else()
        message( "${blue}   Found LAPACK library: ${LAPACK_LIBRARIES}${plain}" )
    endif()
else()
    message( "${red}   LAPACK library not found.${plain}" )
endif()

message( DEBUG "
LAPACK_FOUND        = '${LAPACK_FOUND}'
LAPACK_LIBRARIES    = '${LAPACK_LIBRARIES}'
lapackpp_defs_        = '${lapackpp_defs_}'
")
message( "" )
