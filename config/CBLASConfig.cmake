message(STATUS "Checking for CBLAS...")

string(ASCII 27 Esc)
set(Red         "${Esc}[31m")
set(Blue        "${Esc}[34m")
set(ColourReset "${Esc}[m")

set(CBLAS_DEFINES "")
if(NOT "${MKL_DEFINES}" STREQUAL "")
    set(local_MKL_DEFINES "-D${MKL_DEFINES}")
else()
    set(local_MKL_DEFINES "")
endif()
#message("local_MKL_DEFINES: " ${local_MKL_DEFINES})

try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
    SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/config/cblas.cc
    LINK_LIBRARIES
        ${BLAS_cxx_flags}
        ${BLAS_links}
    COMPILE_DEFINITIONS
        ${local_MKL_DEFINES}
        ${BLAS_int}
    COMPILE_OUTPUT_VARIABLE
        compile_OUTPUT1
    RUN_OUTPUT_VARIABLE
        run_output1
    )

if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
    message("${Blue}  Found CBLAS${ColourReset}")
    set(CBLAS_DEFINES "HAVE_CBLAS")
else()
    message("${Red}  CBLAS not found.${ColourReset}")
endif()

set(run_res1 "")
set(compile_res1 "")
set(run_output1 "")

#message("cblas defines: " ${CBLAS_DEFINES})
