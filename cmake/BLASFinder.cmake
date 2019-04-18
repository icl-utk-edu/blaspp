#message("blas config found: " ${blas_config_found})
if(blas_config_found STREQUAL "TRUE")
    message("BLAS configuration already done!")
    return()
endif()

if(NO_COLOR)
    string(ASCII 27 Esc)
    set(Red         "")
    set(Blue        "")
    set(ColourReset "")
else()
    string(ASCII 27 Esc)
    set(Red         "${Esc}[31m")
    set(Blue        "${Esc}[34m")
    set(ColourReset "${Esc}[m")
endif()

message(STATUS "Looking for BLAS libraries and options")
message(STATUS "Configuring BLAS Fortran mangling...")

set(fortran_mangling
    "-DFORTRAN_ADD_ -DADD_"
    "-DFORTRAN_LOWER -DNOCHANGE"
    "-DFORTRAN_UPPER -DUPCASE"
    )

set(fortran_mangling_names
    "Fortran ADD_"
    "Fortran LOWER"
    "Fortran UPPER"
    )

set(fortran_mangling_clean
    "FORTRAN_ADD_ -DADD_"
    "FORTRAN_LOWER -DNOCHANGE"
    "FORTRAN_UPPER -DUPCASE"
    )
set(FORTRAN_MANGLING_DEFINES "")
list(LENGTH fortran_mangling fort_list_len)

set(BLAS_int_size_names
    "32-bit index array data type"
    "64-bit index array data type"
    )

set(BLAS_int_size_defines
    " "
    "-DLAPACK_ILP64 -DBLAS_ILP64"
    )

set(BLAS_int_size_clean
    " "
    "LAPACK_ILP64 -DBLAS_ILP64"
    )

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
        message("    Fortran convention: " ${mangling_name}${ColourReset})

        LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
        set(BLAS_DEFINES "HAVE_BLAS")
        set(config_found "TRUE")

        break()
    else()
        message("  ${Red}No${ColourReset}")
    endif()

    set(run_res1 "")
    set(compile_res1 "")
    set(run_output1 "")

    math(EXPR j "${j}+1")
endforeach ()

message(STATUS "Checking for BLAS libraries...")

#default_libs   = ['default', 'mkl', 'openblas', 'essl', 'acml', 'accelerate', 'blas']
#default_int    = ['lp64', 'ilp64']
#default_thread = ['sequential', 'threaded']

set(def_lib_list "default;mkl;openblas;essl;acml;accelerate;blas")
set(def_int_list "lp64;ilp64")
set(def_thread_list "sequential;threaded")

set(BLAS_name_list "")
set(BLAS_flag_list "")
set(BLAS_lib_list "")

macro(list_contains var value)
    set(${var})
    foreach(val2 ${ARGN})
        if(${value} STREQUAL ${val2})
            set(${var} TRUE)
        endif()
    endforeach()
endmacro()

function(print_list)
    message("blas_name_list: ${BLAS_name_list}")
    message("blas_flag_list: ${BLAS_flag_list}")
    message("blas_lib_list: ${BLAS_lib_list}")
endfunction()

#set(blas_list "default;mkl;accelerate")
if(BLAS_LIBRARIES)
    set(blas_list ${BLAS_LIBRARIES})
    list(APPEND BLAS_name_list "User supplied")
    list(APPEND BLAS_flag_list "x")
    list(APPEND BLAS_lib_list ${BLAS_LIBRARIES})

    if(OpenMP_CXX_FOUND)
    list(APPEND BLAS_name_list "User supplied with OpenMP")
    list(APPEND BLAS_flag_list ${OpenMP_CXX_FLAGS})
    list(APPEND BLAS_lib_list ${BLAS_LIBRARIES})
    endif()
else()
    set(blas_list ${def_lib_list})
endif()

#message("blas_list: ${blas_list}")
#print_list()

list_contains(does_contain mkl ${blas_list})
if(does_contain)
    message("** Adding mkl to blas list")

    set(mkl_def_int_flag "x;-DMKL_ILP64")
    set(mkl_int_list "")
    set(mkl_int_flag_list "")

    # threaded
    #if (compiler is GNU && compiler using OpenMP):
    #    try -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core
    if(OpenMP_CXX_FOUND)
        set(OpenMP_lib_str "-DLINK_LIBRARIES=OpenMP::OpenMP_CXX")
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
            message("Trying GNU compiler with openmp!")
            list(APPEND BLAS_name_list "Intel MKL lp64, GNU threads (gomp), gfortran")
            list(APPEND BLAS_flag_list ${OpenMP_CXX_FLAGS})
            list(APPEND BLAS_lib_list "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread")

            #message("found GNU compiler with openmp!")
            list(APPEND BLAS_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
            list(APPEND BLAS_flag_list "${OpenMP_CXX_FLAGS} -DMKL_ILP64")
            list(APPEND BLAS_lib_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread")
        #else if (compiler is Intel && compiler using OpenMP):
        #    try -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core
        elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            #message("found icpc and openmp: use Intel threads")
            # icpc -fopenmp implies -liomp5: try mkl_intel_thread, NOT mkl_gnu_thread
            message("Trying Intel compiler with intel threads!")
            list(APPEND BLAS_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "${OpenMP_CXX_FLAGS}")
            list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread")

            #message("found Intel compiler with openmp!")
            list(APPEND BLAS_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "${OpenMP_CXX_FLAGS} -DMKL_ILP64")
            list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread")
        endif()
    else()
        #set(OpenMP_lib_str "")
        #else if (not using OpenMP):
        #    if (compiler is Intel):
        #        try -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
        if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            message("Trying Intel compiler without openmp, try both intel and GUN")
            list(APPEND BLAS_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "-fiomp5")
            list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread")

            list(APPEND BLAS_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "-fiomp5 -DMKL_ILP64")
            list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread")
        endif()

        #    try -lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp
        message("Trying some compiler without openmp!")
        list(APPEND BLAS_name_list "Intel MKL lp64, GNU threads (gomp), gfortran")
        list(APPEND BLAS_flag_list "-lgomp")
        list(APPEND BLAS_lib_list "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread")

        list(APPEND BLAS_name_list "Intel MKL ilp64, GNU threads (gomp), gfortran")
        list(APPEND BLAS_flag_list "-lgomp -DMKL_ILP64")
        list(APPEND BLAS_lib_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread")

        #    if (not compiler is Intel):
        #        try -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5
        if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
            message("Trying non-Intel compiler without openmp!")
            list(APPEND BLAS_name_list "Intel MKL lp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "-fiomp5")
            list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5")

            list(APPEND BLAS_name_list "Intel MKL ilp64, Intel threads (iomp5), ifort")
            list(APPEND BLAS_flag_list "-fiomp5 -DMKL_ILP64")
            list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5")
        endif()
    endif()
    # sequential
    #if (compiler is Intel):
    #    try -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message("Trying Intel compiler without openmp!")
        list(APPEND BLAS_name_list "Intel MKL lp64, sequential, ifort")
        list(APPEND BLAS_flag_list "x")
        list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")

        list(APPEND BLAS_name_list "Intel MKL ilp64, sequential, ifort")
        list(APPEND BLAS_flag_list "-DMKL_ILP64")
        list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")
    endif()

    #try -lmkl_gf_lp64 -lmkl_sequential -lmkl_core
    message("Trying some compiler without openmp!")
    list(APPEND BLAS_name_list "Intel MKL lp64, sequential, gfortran")
    list(APPEND BLAS_flag_list "x")
    list(APPEND BLAS_lib_list "-lmkl_gf_lp64 -lmkl_sequential -lmkl_core")

    list(APPEND BLAS_name_list "Intel MKL ilp64, sequential, gfortran")
    list(APPEND BLAS_flag_list "-DMKL_ILP64")
    list(APPEND BLAS_lib_list "-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core")

    #if (not compiler is Intel):
    #    try -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
    if(NOT "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        message("Trying non-Intel compiler without openmp!")
        list(APPEND BLAS_name_list "Intel MKL lp64, sequential, ifort")
        list(APPEND BLAS_flag_list "x")
        list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core")

        list(APPEND BLAS_name_list "Intel MKL ilp64, sequential, ifort")
        list(APPEND BLAS_flag_list "-DMKL_ILP64")
        list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core")
    endif()

    #print_list()
endif()

if(not_tested)
list_contains(does_contain acml ${blas_list})
if(does_contain)
    list(APPEND BLAS_name_list "AMD ACML threaded")
    list(APPEND BLAS_lib_list "-lacml_mp")
    list(APPEND BLAS_flag_list "x")

    list(APPEND BLAS_name_list "AMD ACML sequential")
    list(APPEND BLAS_lib_list "-lacml")
    list(APPEND BLAS_flag_list "x")
    set(does_contain "")
endif()

list_contains(does_contain essl ${blas_list})
if(does_contain)
    list(APPEND BLAS_name_list "IBM ESSL")
    list(APPEND BLAS_lib_list "-lessl")
    list(APPEND BLAS_flag_list "x")
    set(does_contain "")
endif()

list_contains(does_contain openblas ${blas_list})
if(does_contain)
    list(APPEND BLAS_name_list "OpenBLAS")
    list(APPEND BLAS_lib_list "-lopenblas")
    list(APPEND BLAS_flag_list "x")
    set(does_contain "")
endif()
endif()

set(does_contain "")
list_contains(does_contain accelerate ${blas_list})
if(does_contain)
    message("** Adding default to blas list")
    list(APPEND BLAS_name_list "Apple Accelerate")
    list(APPEND BLAS_flag_list "x")
    list(APPEND BLAS_lib_list "-framework Accelerate")
    set(does_contain "")

    #print_list()
endif()

list_contains(does_contain default ${blas_list})
if(does_contain)
    message("** Adding default to blas list")
    list(APPEND BLAS_name_list "blas")
    list(APPEND BLAS_flag_list "x")
    list(APPEND BLAS_lib_list "-lblas")
    set(does_contain "")

    #print_list()
endif()

set(success_list "")
set(i 0)
foreach(blas_name ${BLAS_name_list})
    #message("i: ${i}")
    list(GET BLAS_flag_list ${i} flag_var)
    list(GET BLAS_lib_list ${i} lib_var)
    if(${flag_var} STREQUAL "x")
        set(flag_var "")
    endif()
    if(${lib_var} STREQUAL "x")
        set(lib_var "")
    endif()
    message("Trying: ${blas_name}")
    message("  flag: ${flag_var}")
    message("   lib: ${lib_var}")

    set(run_result "")
    set(compile_result "")
    set(run_output "")
    set(compile_output "")

    set(j 0)
    foreach(fortran_name ${fortran_mangling_names})
        #message("j: ${j}")
        list(GET fortran_mangling ${j} fort_var)

        try_run(run_result compile_result ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
            LINK_LIBRARIES
                ${lib_var}
                ${flag_var}
            COMPILE_DEFINITIONS
                ${flag_var}
                ${fort_var}
                #${int_define_var}
                #${int_size_var}
            CMAKE_FLAGS
                ${OpenMP_lib_str}
            COMPILE_OUTPUT_VARIABLE
                compile_output
            RUN_OUTPUT_VARIABLE
                run_output
        )

        #message("compile_output: ${compile_output}")
        #message("compile_result: ${compile_result}")
        #message("run_result: ${run_result}")
        #message("run_output: ${run_output}")

        if(compile_result AND NOT "${run_result}" STREQUAL "FAILED_TO_RUN")
            message("${Blue}  SUCCESSFUL compilation${ColourReset}")
            list(APPEND success_list "${blas_name}")

            message("${Blue}  Found working configuration:")

            LIST(GET fortran_mangling ${j} mangling_name)

            message("  Fortran convention: " ${mangling_name})
            message("  BLAS options: " ${lib_var})
            message("  CXX flags: " ${flag_var})
            message("${ColourReset}")
            # part of flags now
            #message("  Integer type:  ${int_define_var}")
            #message("  Integer size:  ${int_size_var}${ColourReset}")

            LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
            set(BLAS_DEFINES "HAVE_BLAS" CACHE INTERNAL "")
            set(config_found "TRUE")

            if(OpenMP_CXX_FOUND)
                set(BLAS_links "${lib_var} ${openmp_flag}" CACHE INTERNAL "")
            else()
                set(BLAS_links "${lib_var}" CACHE INTERNAL "")
            endif()
            set(BLAS_cxx_flags "${flag_var} ${fort_var}" CACHE INTERNAL "")
            string(STRIP ${BLAS_cxx_flags} BLAS_cxx_flags)
            string(STRIP ${BLAS_links} BLAS_links)

            # Break out of MKL checks if we found a working config
            break()
        #else()
            #message("${Red}  Failed compilation${ColourReset}")
        endif()

        math(EXPR j "${j}+1")
        if(BLAS_DEFINES STREQUAL "HAVE_BLAS")
            break()
        endif()
    endforeach()
    math(EXPR i "${i}+1")
    if(BLAS_DEFINES STREQUAL "HAVE_BLAS")
        break()
    endif()
    message("${Red}  Failed compilation${ColourReset}")
endforeach()

if(NOT "${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
    message("${Red}  FAILED TO FIND BLAS LIBRARY${ColourReset}")
    return()
endif()

#message("1 - blas libraries: " ${BLAS_LIBRARIES})
set(BLAS_LIBRARIES ${BLAS_links})
#message("2 - blas libraries: " ${BLAS_LIBRARIES})

if(DEBUG)
    #message("lib defines: " ${LIB_DEFINES})
    message("blas defines: " ${BLAS_DEFINES})
    message("blas libraries: " ${BLAS_LIBRARIES})
    message("cxx flags: " ${BLAS_cxx_flags})
    message("mkl int defines: " ${BLAS_int})
    message("fortran mangling defines: " ${FORTRAN_MANGLING_DEFINES})
    #message("blas complex return: " ${BLAS_RETURN})
    message("config_found: " ${config_found})
endif()

#if(config_found STREQUAL "TRUE")
#    set(blas_config_found "TRUE")
#    message("FOUND BLAS CONFIG")
#endif()
