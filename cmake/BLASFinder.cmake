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

if (ibm_compiler)
    set( fortran_mangling_list
        "-DFORTRAN_LOWER -DNOCHANGE"
        "-DFORTRAN_ADD_ -DADD_"
        "-DFORTRAN_UPPER -DUPCASE"
    )

    set( fortran_mangling_names
        "Fortran LOWER"
        "Fortran ADD_"
        "Fortran UPPER"
    )

    set( fortran_mangling_clean
        "FORTRAN_LOWER -DNOCHANGE"
        "FORTRAN_ADD_ -DADD_"
        "FORTRAN_UPPER -DUPCASE"
    )
else()
    set( fortran_mangling_list
        "-DFORTRAN_ADD_ -DADD_"
        "-DFORTRAN_LOWER -DNOCHANGE"
        "-DFORTRAN_UPPER -DUPCASE"
    )

    set( fortran_mangling_names
        "Fortran ADD_"
        "Fortran LOWER"
        "Fortran UPPER"
    )

    set( fortran_mangling_clean
        "FORTRAN_ADD_ -DADD_"
        "FORTRAN_LOWER -DNOCHANGE"
        "FORTRAN_UPPER -DUPCASE"
    )
endif()
set( fortran_mangling "" )
list( LENGTH fortran_mangling_list fort_list_len )

set( blas_int_size_names
    "32-bit index array data type"
    "64-bit index array data type"
)

set( blas_int_size_defines
    " "
    "-DLAPACK_ILP64 -DBLAS_ILP64"
)

set( blas_int_size_clean
    " "
    "LAPACK_ILP64 -DBLAS_ILP64"
)

set( config_found "" )

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
endif()

#---------------------------------------- default; Cray libsci
if (test_default)
    list( APPEND blas_name_list "default (no library)" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list " " )
endif()

#---------------------------------------- Intel MKL
if (test_mkl)
    # todo: MKL_?(ROOT|DIR)
    if (test_threaded)
        if (OpenMP_CXX_FOUND)
            if (test_gfortran AND gnu_compiler)
                # GNU compiler + OpenMP: require gnu_thread library.
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64, GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS}" )
                    list( APPEND blas_libs_list "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core" )
                endif()

                if (test_int64)
                    list( APPEND blas_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS} -DMKL_ILP64" )
                    list( APPEND blas_libs_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core" )
                endif()

            elseif (test_ifort AND intel_compiler)
                # Intel compiler + OpenMP: require intel_thread library.
                if (test_int)
                    list( APPEND blas_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list "${OpenMP_CXX_FLAGS}" )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core" )
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
                    list( APPEND blas_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
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
                    list( APPEND blas_name_list "Intel MKL lp64, GNU threads (gomp), gfortran")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread" )
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
                    list( APPEND blas_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
                    list( APPEND blas_flag_list " " )
                    list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread" )
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
                list( APPEND blas_name_list "Intel MKL lp64, sequential, ifort" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" )
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
                list( APPEND blas_name_list "Intel MKL lp64, sequential, gfortran" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_gf_lp64 -lmkl_sequential -lmkl_core" )
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
                list( APPEND blas_name_list "Intel MKL lp64, sequential, ifort" )
                list( APPEND blas_flag_list " " )
                list( APPEND blas_libs_list "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core" )
            endif()

            if (test_int64)
                list( APPEND blas_name_list "Intel MKL ilp64, sequential, ifort" )
                list( APPEND blas_flag_list "-DMKL_ILP64" )
                list( APPEND blas_libs_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core" )
            endif()
        endif()  # ifort && not intel
    endif()  # sequential
endif()  # MKL

#---------------------------------------- IBM ESSL
if (test_essl)
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
    endif()

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
    endif()
endif()

#---------------------------------------- OpenBLAS
if (test_openblas)
    # todo: OPENBLAS_?(ROOT|DIR)
    list( APPEND blas_name_list "OpenBLAS" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-lopenblas" )
endif()

#---------------------------------------- Apple Accelerate
if (test_accelerate)
    list( APPEND blas_name_list "Apple Accelerate" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-framework Accelerate" )
endif()

#---------------------------------------- AMD ACML
if (test_acml)
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
endif()

#---------------------------------------- generic -lblas
if (test_generic)
    list( APPEND blas_name_list "generic" )
    list( APPEND blas_flag_list " " )
    list( APPEND blas_libs_list "-lblas" )
endif()

#-------------------------------------------------------------------------------
# Check libraries.

unset( blas_defines CACHE )
set( i 0 )
foreach( blas_name ${blas_name_list} )
    #message( "i: ${i}" )
    list( GET blas_flag_list ${i} blas_flag )
    list( GET blas_libs_list ${i} blas_libs )

    # Undo escaping ; semi-colons to make list.
    string(REPLACE "\;" ";" blas_libs "${blas_libs}")

    message( "${blas_name}" )
    message( "   libs: ${blas_libs}" )
    message( "   flag: ${blas_flag}" )

    set( run_result "" )
    set( compile_result "" )
    set( run_output "" )
    set( compile_output "" )

    set( j 0 )
    foreach( fortran_name ${fortran_mangling_names} )
        #message( "j: ${j}" )
        list( GET fortran_mangling_list ${j} mangling )

        message( "   Fortran: ${mangling}" )
        try_run(
            run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc"
            LINK_LIBRARIES
                "${blas_libs}"
            COMPILE_DEFINITIONS
                "${blas_flag}"
                "${mangling}"
            CMAKE_FLAGS
                "${OpenMP_lib_str}"
            COMPILE_OUTPUT_VARIABLE
                compile_output
            RUN_OUTPUT_VARIABLE
                run_output
        )

        if (compile_result AND "${run_output}" MATCHES "ok")
            message( "${blue}   yes ${default_color}" )
            #message( "${blue}  Found working configuration:" )
            #message( "  BLAS libraries:     ${blas_libs}" )
            #message( "  CXX flags:          ${blas_flag}" )
            #message( "  Fortran convention: ${mangling}" )
            #message( "${default_color}" )

            #LIST( GET fortran_mangling_clean ${j} fortran_mangling )
            set( blas_defines "HAVE_BLAS" CACHE INTERNAL "" )
            set( config_found true )

            if (OpenMP_CXX_FOUND)
                set( blas_links "${blas_libs} ${openmp_flag}" CACHE INTERNAL "" )
            else()
                set( blas_links "${blas_libs}" CACHE INTERNAL "" )
            endif()
            set( blas_cxx_flags "${blas_flag} ${mangling}" CACHE INTERNAL "" )
            string( STRIP "${blas_cxx_flags}" blas_cxx_flags )
            string( STRIP "${blas_links}" blas_links )
            break()
        else()
            message( "${red}   no ${default_color}" )
        endif()

        math( EXPR j "${j}+1" )
        if (blas_defines STREQUAL "HAVE_BLAS")
            break()
        endif()
    endforeach()
    math( EXPR i "${i}+1" )
    if (blas_defines STREQUAL "HAVE_BLAS")
        break()
    endif()
endforeach()

if (NOT "${blas_defines}" STREQUAL "HAVE_BLAS")
    message( "${red}Failed to find BLAS library${default_color}" )
    return()
endif()

set( BLAS_LIBRARIES ${blas_links} )

if (DEBUG)
    #message( "lib defines:     ${lib_defines}" )
    message( "blas defines:     ${blas_defines}" )
    message( "blas libraries:   ${BLAS_LIBRARIES}" )
    message( "cxx flags:        ${blas_cxx_flags}" )
    message( "mkl int defines:  ${blas_int_defines}" )
    message( "fortran mangling: ${fortran_mangling}" )
    #message( "blas complex return:  ${blas_return}" )
    message( "config_found:  ${config_found}" )
endif()
