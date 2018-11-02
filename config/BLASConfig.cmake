string(ASCII 27 Esc)
set(Red         "${Esc}[31m")
set(Blue        "${Esc}[34m")
set(ColourReset "${Esc}[m")

message(STATUS "Checking for BLAS libraries and options")
message(STATUS "Configuring BLAS Fortran mangling and int size...")

set(fortran_mangling
    "-DFORTRAN_ADD_ -DADD_"
    "-DFORTRAN_LOWER -DNOCHANGE"
    "-DFORTRAN_UPPER -DUPCASE"
    )

set(fortran_mangling_names
    "Fortran ADD"
    "Fortran LOWER"
    "Fortran UPPER"
    )

set(fortran_mangling_clean
    "FORTRAN_ADD_ -DADD_"
    "FORTRAN_LOWER -DNOCHANGE"
    "FORTRAN_UPPER -DUPCASE"
    )

set(FORTRAN_MANGLING_DEFINES "")
set(config_found "")

set(j 0)
foreach(fortran_name ${fortran_mangling_names})
    list(GET fortran_mangling ${j} fort_var)

    message ("  ${j} - Trying: ${fortran_name}")
    message ("  ${j}: ${fort_var}")

    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
        COMPILE_DEFINITIONS
            ${fort_var}
        COMPILE_OUTPUT_VARIABLE
            compile_OUTPUT1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        LIST(GET fortran_mangling_name ${j} mangling_name)
        message("  ${Blue}Found valid configuration:")
        message("    Fortran convention: " ${mangling_name})
        message(${ColourReset})

        LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
        set(BLAS_DEFINES "HAVE_BLAS")
        set(config_found "TRUE")

        break()
    else()
        message("  ${Red}No")
        message(${ColourReset})
    endif()

    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")

    math(EXPR j "${j}+1")
endforeach ()

message(STATUS "Configuring for BLAS libraries...")

set(BLAS_int_defines_names
    "32-bit index array data type"
    "64-bit index array data type"
    )

set(BLAS_int_defines
    "-DLAPACK_LP64 -DBLAS_LP64"
    "-DMKL_ILP64 -DLAPACK_ILP64 -DBLAS_ILP64"
    )

set(BLAS_int_defines_clean
    "LAPACK_LP64 -DBLAS_LP64"
    "MKL_ILP64 -DLAPACK_ILP64 -DBLAS_ILP64"
    )

set(BLAS_lib_options
    # each pair has Intel conventions, then GNU conventions

    # int, threaded
    #['Intel MKL (int, Intel conventions, threaded)',
    "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm"
    #['Intel MKL (int, GNU conventions, threaded)',
    "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm"

    # int64_t, threaded
    #['Intel MKL (int64_t, Intel conventions, threaded)',
    "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm"
    #['Intel MKL (int64_t, GNU conventions, threaded)',
    "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm"

    # int, sequential
    #['Intel MKL (int, Intel conventions, sequential)',
    "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm"
    #['Intel MKL (int, GNU conventions, sequential)',
    "-lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm"

    # int64_t, sequential
    #['Intel MKL (int64_t, Intel conventions, sequential)',
    "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm"
    #['Intel MKL (int64_t, GNU conventions, sequential)',
    "-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core -lm"
)

list(LENGTH BLAS_lib_options blas_list_len)

set(BLAS_cxx_flags_flags
    "-fopenmp"
    "-fopenmp"
    "-fopenmp"
    "-fopenmp"
    ""
    ""
    ""
    ""
    )

set(BLAS_int_definition_list
    ""
    ""
    "-DMKL_ILP64"
    "-DMKL_ILP64"
    ""
    ""
    "-DMKL_ILP64"
    "-DMKL_ILP64"
    )

set(BLAS_names
    "Intel MKL (int, Intel conventions, threaded)"
    "Intel MKL (int, GNU conventions, threaded)"
    "Intel MKL (int64_t, Intel conventions, threaded)"
    "Intel MKL (int64_t, GNU conventions, threaded)"
    "Intel MKL (int, Intel conventions, sequential)"
    "Intel MKL (int, GNU conventions, sequential)"
    "Intel MKL (int64_t, Intel conventions, sequential)"
    "Intel MKL (int64_t, GNU conventions, sequential)"
    )

set(BLAS_INT_DEFINES "")
set(BLAS_DEFINES "")
set(MKL_DEFINES "")
set(BLAS_links "")
set(BLAS_cxx_flags "")
set(BLAS_int "")

set(j 0)
foreach(fortran_name ${fortran_mangling_names})
    set(i 0)
    foreach (lib_name ${BLAS_names})
        list(GET fortran_mangling ${j} fort_var)
        list(GET BLAS_lib_options ${i} lib_var)
        list(GET BLAS_cxx_flags_flags ${i} cxx_flag)
        list(GET BLAS_int_definition_list ${i} int_define_var)

        message ("  ${j},${i} - Trying: ${fortran_name} ${lib_name}")

        try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
            LINK_LIBRARIES
                ${lib_var}
                ${cxx_flag}
            COMPILE_DEFINITIONS
                ${fort_var}
                ${int_define_var}
            COMPILE_OUTPUT_VARIABLE
                compile_OUTPUT1
            RUN_OUTPUT_VARIABLE
                run_output1
        )

        if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
            message("${Blue}  Found working configuration:")

            #LIST(GET BLAS_int_defines_names ${i} int_name)
            LIST(GET fortran_mangling ${j} mangling_name)

            message("  Fortran convention: " ${mangling_name})
            message("  BLAS options: " ${lib_var})
            message("  CXX flags: " ${cxx_flag})
            message("  Integer type:  ${int_define_var}")
            message("${ColourReset}")

            LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
            set(BLAS_DEFINES "HAVE_BLAS")
            set(config_found "TRUE")

            set(BLAS_links ${lib_var})
            set(BLAS_cxx_flags ${cxx_flag})
            set(BLAS_int ${int_define_var})

            break()
        else()
            message("${Red}  FAIL${ColourReset} at (${j},${i})")
        endif()

        set(run_res1 "")
        set(compile_res1 "")
        set(run_output1 "")

        math(EXPR i "${i}+1")
        if(NOT (i LESS blas_list_len))
            break()
        endif()
    endforeach ()
    if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        break()
    endif()
    math(EXPR j "${j}+1")
    if(config_found STREQUAL "TRUE")
        message("${Red}  FAILED TO FIND BLAS CONFIG${ColourReset}")
        break()
    endif()
endforeach ()

if(${BLAS_DEFINES} MATCHES "HAVE_BLAS")
    message(STATUS "Checking for MKL version number...")

    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/mkl_version.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_OUTPUT1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        message("${Blue}  ${run_output1}${ColourReset}")
        set(MKL_DEFINES "HAVE_MKL")
    else()
        message(FATAL_ERROR "${Red}  MKL was found, but version was not determined${ColourReset}")
    endif()
endif()

message(STATUS "Checking BLAS complex return type...")

try_run(run_res1
    compile_res1
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex_argument.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${BLAS_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  BLAS (zdotc) returns complex as hidden argument (Intel ifort convention)${ColourReset}")
    set(BLAS_RETURN "BLAS_COMPLEX_RETURN_ARGUMENT")
else()
    message("${Blue}  BLAS (zdotc) returns complex (GNU gfortran convention) - no extra definitions needed${ColourReset}")
    set(BLAS_RETURN "")
endif()

if(DEBUG)
message("mkl defines: " ${MKL_DEFINES})
message("blas defines: " ${BLAS_DEFINES})
message("mkl int defines: " ${BLAS_INT_DEFINES})
message("fortran mangling defines: " ${FORTRAN_MANGLING_DEFINES})
message("blas complex return: " ${BLAS_RETURN})
endif()
