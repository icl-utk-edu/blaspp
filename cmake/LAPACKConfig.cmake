if(NOT ${LAPACK_DEFINES} STREQUAL "")
    message("CBLAS configuration already done!")
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

set(local_mangling "-D${FORTRAN_MANGLING_DEFINES}")
set(local_int "-D${BLAS_INT_DEFINES}")

if(NOT "${LIB_DEFINES}" STREQUAL "")
    set(local_mkl_defines "-D${LIB_DEFINES}")
else()
    set(local_mkl_defines "")
endif()
if(NOT "${BLAS_DEFINES}" STREQUAL "")
    set(local_blas_defines "-D${BLAS_DEFINES}")
else()
    set(local_blas_defines "")
endif()
if(NOT "${BLAS_INT_DEFINES}" STREQUAL "")
    set(local_int "-D${BLAS_INT_DEFINES}")
else()
    set(local_int "")
endif()

message(STATUS "Checking for LAPACK POTRF...")

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_potrf.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_output1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

#message("compile_output: ${compile_output1}")

# if it compiled and ran, then LAPACK is available
if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  Found LAPACK${ColourReset}")
    set(LAPACK_DEFINES "HAVE_LAPACK" CACHE INTERNAL "")
else()
    message("${Red}  LAPACK not found${ColourReset}")
    set(LAPACK_DEFINES "" CACHE INTERNAL "")
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")
return()
message(STATUS "Checking for LAPACKE POTRF...")

try_run(run_res1 compile_res1
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapacke_potrf.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  Found LAPACKE${ColourReset}")
    set(LAPACKE_DEFINES "HAVE_LAPACKE")
else()
    message("${Red}  FAIL${ColourReset} at (${j},${i})")
    set(LAPACKE_DEFINES "")
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

message(STATUS "Checking for XBLAS...")

try_run(run_res1 compile_res1
    ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_xblas.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  Found XBLAS${ColourReset}")
    set(XBLAS_DEFINES "HAVE_XBLAS")
else()
    message("${Red}  XBLAS not found.${ColourReset}")
    set(XBLAS_DEFINES "")
endif()
set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

message(STATUS "Checking LAPACK version...")

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/lapack_version.cc
    LINK_LIBRARIES
        ${BLAS_links}
        ${BLAS_cxx_flags}
    COMPILE_DEFINITIONS
        ${local_mkl_defines}
        ${local_blas_defines}
        ${local_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

if (compile_res1 AND "${run_output1}" MATCHES "ok")
    message("${Blue}  Found LAPACK version number.${ColourReset}")

    string(REPLACE "=" ";" run_out_list ${run_output1})
    list(GET run_out_list 1 version_number)
    string(REPLACE "../config" ";" version_list ${version_number})

    list(GET version_list 0 major_ver)
    list(GET version_list 1 minor_ver)
    list(GET version_list 2 rev_ver)

    # For some reason, the version number strings have extra characters, remove.
    string(REGEX REPLACE "[^0-9]" "" minor_ver ${minor_ver})
    string(LENGTH ${minor_ver} minor_len)
    if(minor_len LESS 2)
        set(minor_ver "0${minor_ver}")
    endif()

    # Remove extra characters.
    string(REGEX REPLACE "[^0-9]" "" rev_ver ${rev_ver})
    string(LENGTH ${rev_ver} rev_len)
    if(rev_len LESS 2)
        set(rev_ver "0${rev_ver}")
    endif()

    set(LAPACK_VER_DEFINE "LAPACK_VERSION=${major_ver}${minor_ver}${rev_ver}")
else()
    message("${Red}  Failed to determine LAPACK version.${ColourReset}")
    set(LAPACK_VER_DEFINE "")
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

#message("lapack defines: " ${LAPACK_DEFINES})
#message("lapacke defines: " ${LAPACKE_DEFINES})
#message("xblas defines: " ${XBLAS_DEFINES})
#message("lapack version define: " ${LAPACK_VER_DEFINE})
