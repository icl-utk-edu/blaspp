set(MKL_names
        "Intel MKL (int, Intel conventions, threaded)"
        "Intel MKL (int, GNU conventions, threaded)"
        "Intel MKL (int64_t, Intel conventions, threaded)"
        "Intel MKL (int64_t, GNU conventions, threaded)"
        "Intel MKL (int, Intel conventions, sequential)"
        "Intel MKL (int, GNU conventions, sequential)"
        "Intel MKL (int64_t, Intel conventions, sequential)"
        "Intel MKL (int64_t, GNU conventions, sequential)"
        )

set(MKL_flags
        "-fopenmp"
        "-fopenmp"
        "-fopenmp -DMKL_ILP64"
        "-fopenmp -DMKL_ILP64"
        ""
        ""
        "-DMKL_ILP64"
        "-DMKL_ILP64"
        )

#set(MKL_ld_flags
#        "-fopenmp"
#        "-fopenmp"
#        "-fopenmp"
#        "-fopenmp"
#        ""
#        ""
#        "-DMKL_ILP64"
#        "-DMKL_ILP64"
#        )

set(MKL_lib_options
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

string(ASCII 27 Esc)
set(Red         "${Esc}[31m")
set(Blue        "${Esc}[34m")
set(ColourReset "${Esc}[m")

message(STATUS "Checking Intel MKL libraries...")

set(MKL_DEFINES "")

set(i 0)
foreach (name ${MKL_names})
    list(GET BLAS_cxx_flags_flags ${i} cxx_var)
    #list(GET MKL_ld_flags ${i} ld_var)
    list(GET MKL_lib_options ${i} lib_var)
    message ("  ${i}: ${name} ${cxx_var} ${ld_var} ${lib_var}")
    #message ("  ${i} - Trying: ${name}")
    message ("          Flags: ${cxx_var}")
    #message ("         Linking: ${lib_var}")

    try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            ${CMAKE_CURRENT_SOURCE_DIR}/config/mkl_version.cc
        LINK_LIBRARIES
            ${lib_var}
            ${cxx_var}
        #COMPILE_DEFINITIONS
            #${ld_var}
        COMPILE_OUTPUT_VARIABLE
            compile_OUTPUT1
        RUN_OUTPUT_VARIABLE
            run_output1
        )
    #MESSAGE("run_res1: ${run_res1}")
    #MESSAGE("compile_res1: ${compile_res1}")
    #message("run_output: ${run_output1}")
    #message("compile_output: ${compile_OUTPUT1}")

    if (compile_res1)
      message("${Blue}  Found working configuration:")
      message("    " ${lib_var})
      message("    " ${cxx_var})
      message("    " ${ld_var}${ColourReset})
      set(MKL_DEFINES "HAVE_MKL")
      break()
    else()
        #message("${Red}  FAIL${ColourReset}")
    endif()

    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")

    math(EXPR i "${i}+1")
endforeach ()

#message("  Working config:")
#message("  " ${lib_var})
#message("  " ${cxx_var})
#message("  " ${ld_var})

set(BLAS_links ${lib_var})
set(BLAS_cxx_flags ${cxx_var})
set(MKL_def ${ld_var})