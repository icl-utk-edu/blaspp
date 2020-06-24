# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#message( "blas config found: ${blas_config_found}" )
#if (blas_config_found STREQUAL "TRUE")
#    message( "BLAS configuration already done!" )
#    return()
#endif()

include( "cmake/util.cmake" )

message( STATUS "Looking for BLAS libraries and options" )

#-------------------------------------------------------------------------------
# Prints the BLAS_{name,flag,libs}_lists.
#
function( debug_print_list msg )
    if ("${CMAKE_MESSAGE_LOG_LEVEL}" MATCHES "DEBUG|TRACE")
        message( "---------- lists: ${msg}" )
        message( "blas_name_list = ${blas_name_list}" )
        message( "blas_flag_list = ${blas_flag_list}" )
        message( "blas_libs_list = ${blas_libs_list}" )

        message( "\nrow;  ${red}blas_name;${plain}  blas_libs;  ${red}blas_flag${plain}" )
        set( i 0 )
        foreach (name IN LISTS blas_name_list)
            list( GET blas_flag_list ${i} flag )
            list( GET blas_libs_list ${i} libs )
            message( "${i};  ${red}${name};${plain}  ${libs};  ${red}${flag}${plain}" )
            math( EXPR i "${i} + 1" )
        endforeach()
        message( "" )
    endif()
endfunction()

#-------------------------------------------------------------------------------
# Setup.

#---------------------------------------- compiler
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set( gnu_compiler true )
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
    set( intel_compiler true )
endif()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "XL")
    set( ibm_compiler true )
endif()

#---------------------------------------- Fortran manglings to test
if (ibm_compiler)
    # For IBM XL, change default mangling search order to lower, add_, upper,
    # ESSL includes all 3, but Netlib LAPACK has only one mangling.
    set( fortran_mangling_list
        "-DFORTRAN_LOWER"
        "-DFORTRAN_ADD_"
        "-DFORTRAN_UPPER"
    )
else()
    # For all others, mangling search order as add_, lower, upper,
    # since add_ is the most common.
    set( fortran_mangling_list
        "-DFORTRAN_ADD_"
        "-DFORTRAN_LOWER"
        "-DFORTRAN_UPPER"
    )
endif()

#---------------------------------------- integer sizes to test
set( int_size_list
    " "             # int (LP64)
    "-DBLAS_ILP64"  # int64_t (ILP64)
)

#-------------------------------------------------------------------------------
# Parse options: BLAS_LIBRARIES, blas, blas_int, blas_threaded, blas_fortran.

set( test_all true )

#---------------------------------------- BLAS_LIBRARIES
if (BLAS_LIBRARIES)
    set( test_blas_libraries true )
    set( test_all false )
endif()

#---------------------------------------- blas
string( TOLOWER "${blas}" blas_ )

if ("${blas_}" MATCHES "acml")
    set( test_acml true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "apple|accelerate")
    set( test_accelerate true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "cray|libsci|default")
    set( test_default true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "ibm|essl")
    set( test_essl true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "intel|mkl")
    set( test_mkl true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "openblas")
    set( test_openblas true )
    set( test_all false )
endif()

if ("${blas_}" MATCHES "generic")
    set( test_generic true )
    set( test_all false )
endif()

message( DEBUG "
BLAS_LIBRARIES      = '${BLAS_LIBRARIES}'
blas                = '${blas}'
blas_               = '${blas_}'
test_blas_libraries = '${test_blas_libraries}'
test_acml           = '${test_acml}'
test_accelerate     = '${test_accelerate}'
test_default        = '${test_default}'
test_essl           = '${test_essl}'
test_mkl            = '${test_mkl}'
test_openblas       = '${test_openblas}'
test_generic        = '${test_generic}'
test_all            = '${test_all}'")

#---------------------------------------- blas_fortran
set( TOLOWER "${blas_fortran}" blas_fortran_ )

if ("${blas_fortran_}" MATCHES "gfortran")
    set( test_gfortran true )
endif()
if ("${blas_fortran_}" MATCHES "ifort")
    set( test_ifort true )
endif()
# Otherwise, test both.
if (NOT (test_gfortran OR test_ifort))
    set( test_gfortran true )
    set( test_ifort    true )
endif()

message( DEBUG "
blas_fortran        = '${blas_fortran}'
blas_fortran_       = '${blas_fortran_}'
test_gfortran       = '${test_gfortran}'
test_ifort          = '${test_ifort}'")

#---------------------------------------- blas_int
set( TOLOWER "${blas_int}" blas_int_ )

# This regex is similar to "\b(lp64|int)\b".
if ("${blas_int_}" MATCHES "(^|[^a-zA-Z0-9_])(lp64|int|int32|int32_t)($|[^a-zA-Z0-9_])")
    set( test_int true )
endif()
if ("${blas_int_}" MATCHES "(^|[^a-zA-Z0-9_])(ilp64|int64|int64_t)($|[^a-zA-Z0-9_])")
    set( test_int64 true )
endif()
# Otherwise, test both.
if (NOT (test_int OR test_int64))
    set( test_int   true )
    set( test_int64 true )
endif()

message( DEBUG "
blas_int            = '${blas_int}'
blas_int_           = '${blas_int_}'
test_int            = '${test_int}'
test_int64          = '${test_int64}'")

#---------------------------------------- blas_threaded
set( TOLOWER "${blas_threaded}" blas_threaded_ )

# This regex is similar to "\b(yes|...)\b".
if ("${blas_threaded_}" MATCHES "(^|[^a-zA-Z0-9_])(yes|true|on|1)($|[^a-zA-Z0-9_])")
    set( test_threaded true )
endif()
if ("${blas_threaded_}" MATCHES "(^|[^a-zA-Z0-9_])(no|false|off|0)($|[^a-zA-Z0-9_])")
    set( test_sequential true )
endif()
# Otherwise, test both.
if (NOT (test_threaded OR test_sequential))
    set( test_threaded   true )
    set( test_sequential true )
endif()

message( DEBUG "
blas_threaded       = '${blas_threaded}'
blas_threaded_      = '${blas_threaded_}'
test_threaded       = '${test_threaded}'
test_sequential     = '${test_sequential}'")

#-------------------------------------------------------------------------------
# Build list of libraries to check.

if (OpenMP_CXX_FOUND)
    set( OpenMP_libs "-DLINK_LIBRARIES=OpenMP::OpenMP_CXX" )
endif()

set( blas_name_list "" )
set( blas_flag_list "" )
set( blas_libs_list "" )

#---------------------------------------- BLAS_LIBRARIES
if (test_blas_libraries)
    # Escape ; semi-colons so we can append it as one item to a list.
    string( REPLACE ";" "\\\;" BLAS_LIBRARIES_ESC "${BLAS_LIBRARIES}" )
    message( DEBUG "BLAS_LIBRARIES ${BLAS_LIBRARIES}" )
    message( DEBUG "   =>          ${BLAS_LIBRARIES_ESC}" )
    message( "..." )

    list( APPEND blas_name_list "\$BLAS_LIBRARIES" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "${BLAS_LIBRARIES_ESC}" )
    debug_print_list( "BLAS_LIBRARIES" )
endif()

#---------------------------------------- default; Cray libsci
if (test_all OR test_default)
    list( APPEND blas_name_list "default (no library)" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list " " )
    debug_print_list( "default" )
endif()

#---------------------------------------- Intel MKL
if (test_all OR test_mkl)
    # todo: MKL_?(ROOT|DIR)
    if (test_threaded)
        if (OpenMP_CXX_FOUND)
            if (test_gfortran AND gnu_compiler)
                # GNU compiler + OpenMP: require gnu_thread library.
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64,  GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS}" )
                    list( APPEND blas_libs_list "-lmkl_gf_lp64  -lmkl_gnu_thread -lmkl_core" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS} -DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core" )
                endif()

            elseif (test_ifort AND intel_compiler)
                # Intel compiler + OpenMP: require intel_thread library.
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64,  Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS}" )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS} -DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core" )
                endif()

            else()
                # MKL doesn't have libraries for other OpenMP backends.
                message( "Skipping threaded MKL for non-GNU, non-Intel compiler with OpenMP" )
            endif()
        else()
            # If Intel compiler, prefer Intel ifort interfaces.
            if (test_ifort AND intel_compiler)
                # Intel compiler, no OpenMP: add -liomp5.
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64,  Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list "-DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
                endif()
            endif()  # ifort

            # Otherwise, prefer GNU gfortran interfaces.
            if (test_gfortran)
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64,  GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_gf_lp64  -lmkl_gnu_thread -lmkl_core -lgomp -lpthread" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list "-DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread" )
                endif()
            endif()  # gfortran

            # Not Intel compiler, lower preference for Intel ifort interfaces.
            # todo: same as Intel block above.
            if (test_ifort AND NOT intel_compiler)
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64,  Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list "-DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
                endif()
            endif()  # ifort && not intel
        endif()
    endif()

    #----------
    if (test_sequential)
        # If Intel compiler, prefer Intel ifort interfaces.
        if (test_ifort AND intel_compiler)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, ifort" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, ifort" )
                list( APPEND blas_flag_list "-DMKL_ILP64" )
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # ifort

        # Otherwise, prefer GNU gfortran interfaces.
        if (test_gfortran)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, gfortran" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_gf_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, gfortran" )
                list( APPEND blas_flag_list "-DMKL_ILP64" )
                list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # gfortran

        # Not Intel compiler, lower preference for Intel ifort interfaces.
        if (test_ifort AND NOT intel_compiler)
            if (test_int)
                list( APPEND blas_name_list "Intel MKL lp64,  sequential, ifort" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_intel_lp64  -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, ifort" )
                list( APPEND blas_flag_list "-DMKL_ILP64" )
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # ifort && not intel
    endif()  # sequential
    debug_print_list( "mkl" )
endif()  # MKL

#---------------------------------------- IBM ESSL
if (test_all OR test_essl)
    # todo: ESSL_?(ROOT|DIR)
    if (test_threaded)
        if (ibm_compiler)
            if (test_int)
                list( APPEND blas_name_list "IBM ESSL int (lp64), multi-threaded"  )
                list( APPEND blas_flag_list " "  )
                list( APPEND blas_libs_list "-lesslsmp -lxlsmp"  )
                # ESSL manual says '-lxlf90_r -lxlfmath' also,
                # but this doesn't work on Summit
            endif()

            if (test_int64)
                list( APPEND blas_name_list "IBM ESSL int64 (ilp64), multi-threaded"  )
                list( APPEND blas_flag_list "-D_ESV6464"  )
                list( APPEND blas_libs_list "-lesslsmp6464 -lxlsmp"  )
            endif()
        elseif (OpenMP_CXX_FOUND)
            if (test_int)
                list( APPEND blas_name_list "IBM ESSL int (lp64), multi-threaded, with OpenMP"  )
                list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS}"  )
                list( APPEND blas_libs_list "-lesslsmp"  )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "IBM ESSL int64 (ilp64), multi-threaded, with OpenMP"  )
                list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS} -D_ESV6464"  )
                list( APPEND blas_libs_list "-lesslsmp6464"  )
            endif()
        endif()
    endif()  # threaded

    if (test_sequential)
        if (test_int)
            list( APPEND blas_name_list "IBM ESSL int (lp64), sequential"  )
            list( APPEND blas_flag_list " "  )
            list( APPEND blas_libs_list "-lessl"  )
        endif()

        if (test_int64)
            list( APPEND blas_name_list "IBM ESSL int64 (ilp64), sequential"  )
            list( APPEND blas_flag_list "-D_ESV6464"  )
            list( APPEND blas_libs_list "-lessl6464"  )
        endif()
    endif()  # sequential
    debug_print_list( "essl" )
endif()

#---------------------------------------- OpenBLAS
if (test_all OR test_openblas)
    # todo: OPENBLAS_?(ROOT|DIR)
    list( APPEND blas_name_list "OpenBLAS" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-lopenblas" )
    debug_print_list( "openblas" )
endif()

#---------------------------------------- Apple Accelerate
if (test_all OR test_accelerate)
    list( APPEND blas_name_list "Apple Accelerate" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-framework Accelerate" )
    debug_print_list( "accelerate" )
endif()

#---------------------------------------- AMD ACML
if (test_all OR test_acml)
    # todo: ACML_?(ROOT|DIR)
    if (test_threaded)
        list( APPEND blas_name_list "AMD ACML threaded" )
        list( APPEND blas_flag_list " " )
        list( APPEND blas_libs_list "-lacml_mp" )
    endif()

    if (test_sequential)
        list( APPEND blas_name_list "AMD ACML sequential" )
        list( APPEND blas_flag_list " " )
        list( APPEND blas_libs_list "-lacml" )
    endif()
    debug_print_list( "acml" )
endif()

#---------------------------------------- generic -lblas
if (test_all OR test_generic)
    list( APPEND blas_name_list "generic" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-lblas" )
    debug_print_list( "generic" )
endif()

#-------------------------------------------------------------------------------
# Check each BLAS library.

unset( BLAS_FOUND CACHE )
unset( blas_defines CACHE )

set( i 0 )
foreach (blas_name IN LISTS blas_name_list)
    message( TRACE "i: ${i}" )
    list( GET blas_flag_list ${i} blas_flag )
    list( GET blas_libs_list ${i} blas_libs )
    math( EXPR i "${i}+1" )

    # Undo escaping ; semi-colons to make list.
    string(REPLACE "\;" ";" blas_libs "${blas_libs}")

    message( "\n${blas_name}" )
    message( "   libs:  ${blas_libs}" )
    message( "   flags: ${blas_flag}" )

    foreach (mangling IN LISTS fortran_mangling_list)
        foreach (int_size IN LISTS int_size_list)
            set( label "   ${mangling} ${int_size}" )
            pad_string( "${label}" 50 label )

            # Try to link a simple hello world with the library.
            try_compile(
                link_result ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                    "${CMAKE_CURRENT_SOURCE_DIR}/config/hello.cc"
                LINK_LIBRARIES
                    "${blas_libs}"
                COMPILE_DEFINITIONS
                    "${blas_flag} ${mangling} ${int_size}"
                CMAKE_FLAGS
                    "${OpenMP_lib_str}"
                COMPILE_OUTPUT_VARIABLE
                    link_output
            )
            debug_try_compile( "hello.cc" "${link_result}" "${link_output}" )

            # If hello didn't link, assume library not found,
            # so break both mangling & int_size loops.
            if (NOT link_result)
                message( "${label} ${red} no (library not found)${plain}" )
                break()
            endif()

            # Try to link and run simple BLAS routine with the library.
            try_run(
                run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                    "${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc"
                LINK_LIBRARIES
                    "${blas_libs}"
                COMPILE_DEFINITIONS
                    "${blas_flag} ${mangling} ${int_size}"
                CMAKE_FLAGS
                    "${OpenMP_lib_str}"
                COMPILE_OUTPUT_VARIABLE
                    compile_output
                RUN_OUTPUT_VARIABLE
                    run_output
            )
            debug_try_run( "blas.cc" "${compile_result}" "${compile_output}"
                                     "${run_result}" "${run_output}" )

            # If int32 didn't link, int64 won't either, so break int_size loop.
            if (NOT compile_result)
                message( "${label} ${red} no (didn't link: routine not found)${plain}" )
                break()
            endif()

            # If it runs (exits 0), we're done, so break all 3 loops.
            if (run_result EQUAL 0 AND run_output MATCHES "ok")
                message( "${label} ${blue} yes${plain}" )

                set( BLAS_FOUND true
                     CACHE INTERNAL "Whether BLAS library was found" )

                set( BLAS_LIBRARIES ${blas_libs}
                     CACHE STRING "" )

                set( blas_defines "${blas_flag} ${mangling} ${int_size}"
                     CACHE INTERNAL "Constants defined for BLAS" )
                break()
            else()
                message( "${label} ${red} no (didn't run: int mismatch, etc.)${plain}" )
            endif()
        endforeach()

        # Break loops as described above.
        if (NOT link_result OR BLAS_FOUND)
            break()
        endif()
    endforeach()

    # Break loops as described above.
    if (BLAS_FOUND)
        break()
    endif()
endforeach()

#-------------------------------------------------------------------------------
if (BLAS_FOUND)
    message( "${blue}   Found BLAS library: ${BLAS_LIBRARIES}.${plain}" )
else()
    message( "${red}   Failed to find a BLAS library.${plain}" )
endif()

message( DEBUG "
BLAS_FOUND:         = '${BLAS_FOUND}'
BLAS_LIBRARIES      = '${BLAS_LIBRARIES}'
blas_defines        = '${blas_defines}'")
