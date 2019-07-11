#message("blas config found: " ${blas_config_found})
if(blas_config_found STREQUAL "TRUE")
    message("BLAS configuration already done!")
    return()
endif()

if(COLOR)
    string(ASCII 27 Esc)
    set(Red         "${Esc}[31m")
    set(Blue        "${Esc}[34m")
    set(ColourReset "${Esc}[m")
else()
    string(ASCII 27 Esc)
    set(Red         "")
    set(Blue        "")
    set(ColourReset "")
endif()

message(STATUS "Checking for BLAS library options")

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")
set(compile_output1 "")

if(${BLAS_DEFINES} MATCHES "HAVE_BLAS")
    #message(STATUS "Checking for library vendors ...")

    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/mkl_version.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND "${run_output1}" MATCHES "MKL_VERSION")
        message("${Blue}  ${run_output1}${ColourReset}")
        set(LIB_DEFINES "HAVE_MKL" CACHE INTERNAL "")
    else()
        set(LIB_DEFINES "")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

if(${BLAS_DEFINES} MATCHES "HAVE_BLAS" AND
   "${LIB_DEFINES}" STREQUAL "")
    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/acml_version.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND "${run_output1}" MATCHES "ok")
        message("${Blue}  ${run_output1}${ColourReset}")
        set(LIB_DEFINES "HAVE_ACML" CACHE INTERNAL "")
    else()
        set(LIB_DEFINES "" CACHE INTERNAL "")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

if(${BLAS_DEFINES} MATCHES "HAVE_BLAS" AND
   "${LIB_DEFINES}" STREQUAL "")
    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/essl_version.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND "${run_output1}" MATCHES "ESSL_VERSION")
        message("${Blue}  ${run_output1}${ColourReset}")
        set(LIB_DEFINES "HAVE_ESSL" CACHE INTERNAL "")
    else()
        set(LIB_DEFINES "" CACHE INTERNAL "")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

if(${BLAS_DEFINES} MATCHES "HAVE_BLAS" AND
    "${LIB_DEFINES}" STREQUAL "")
    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/openblas_version.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND "${run_output1}" MATCHES "ok")
        message("${Blue}  ${run_output1}${ColourReset}")
        set(LIB_DEFINES "HAVE_OPENBLAS" CACHE INTERNAL "")
    else()
        set(LIB_DEFINES "" CACHE INTERNAL "")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

message(STATUS "Checking BLAS complex return type...")

try_run(run_res1
    compile_res1
        ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/return_complex.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${BLAS_int}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

#message ('compile result: ' ${compile_res1})
#message ('run result: ' ${run_res1})
#message ('compile output: ' ${compile_output1})
#message ('run output: ' ${run_output1})

if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  BLAS (zdotc) returns complex (GNU gfortran convention)${ColourReset}")
    set(BLAS_COMPLEX_RETURN "")
else()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")

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
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
        )

    if (compile_res1 AND "${run_output1}" MATCHES "ok")
        message("${Blue}  BLAS (zdotc) returns complex as hidden argument (Intel ifort convention)${ColourReset}")
        set(BLAS_COMPLEX_RETURN "BLAS_COMPLEX_RETURN_ARGUMENT")
    else()
        message(FATAL_ERROR "Error - Cannot detect zdotc return value. Please check the BLAS installation.")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

message(STATUS "Checking BLAS float return type...")

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/return_float.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${BLAS_int}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output1
)

if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  BLAS (sdot) returns float as float (standard)${ColourReset}")
else()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/return_float_f2c.cc
        LINK_LIBRARIES
            ${BLAS_links}
            ${BLAS_cxx_flags}
        COMPILE_DEFINITIONS
            ${BLAS_int}
        COMPILE_OUTPUT_VARIABLE
            compile_output1
        RUN_OUTPUT_VARIABLE
            run_output1
    )

    if (compile_res1 AND "${run_output1}" MATCHES "ok")
        message("${Blue}  BLAS (sdot) returns float as double (f2c convention)${ColourReset}")
        set(BLAS_FLOAT_RETURN "HAVE_F2C")
    endif()
    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")
    set(compile_output1 "")
endif()

if(DEBUG)
    message("lib defines: " ${LIB_DEFINES})
    message("blas defines: " ${BLAS_DEFINES})
    message("mkl int defines: " ${BLAS_int})
    message("fortran mangling defines: " ${FORTRAN_MANGLING_DEFINES})
    message("blas complex return: " ${BLAS_COMPLEX_RETURN})
    message("blas float return: " ${BLAS_FLOAT_RETURN})
    message("config_found: " ${config_found})
endif()

if(config_found STREQUAL "TRUE")
    #set(blas_config_found "TRUE")
    #message("FOUND BLAS CONFIG")
    set (blas_config_found "TRUE" CACHE STRING "Set TRUE if BLAS config is found")
endif()
