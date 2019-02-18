message("blas config found: " ${blas_config_found})
if(blas_config_found STREQUAL "TRUE")
    message("BLAS configuration already done!")
    return()
endif()


string(ASCII 27 Esc)
set(Red         "${Esc}[31m")
set(Blue        "${Esc}[34m")
set(ColourReset "${Esc}[m")

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

message(STATUS "Configuring for BLAS libraries...")

set(BLAS_lib_list "")
set(BLAS_cxx_flag_list "")
set(BLAS_int_definition_list "")

if("${BLAS_LIBRARY}" STREQUAL "auto" OR "${BLAS_LIBRARY}" STREQUAL "Intel MKL")
    #int, Intel conventions, threaded
    list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm")
    list(APPEND BLAS_cxx_flag_list "-fopenmp")
    list(APPEND BLAS_int_definitions_list " ")
    #int, GNU conventions, threaded
    list(APPEND BLAS_lib_list "-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm")
    list(APPEND BLAS_cxx_flag_list "-fopenmp")
    list(APPEND BLAS_int_definitions_list " ")

    #int64_t, Intel conventions, threaded
    list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm")
    list(APPEND BLAS_cxx_flag_list "-fopenmp")
    list(APPEND BLAS_int_definitions_list "-DMKL_ILP64")
    #int64_t, GNU conventions, threaded
    list(APPEND BLAS_lib_list "-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm")
    list(APPEND BLAS_cxx_flag_list "-fopenmp")
    list(APPEND BLAS_int_definitions_list "-DMKL_ILP64")

    #int, Intel conventions, sequential
    list(APPEND BLAS_lib_list "-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm")
    list(APPEND BLAS_cxx_flag_list "")
    list(APPEND BLAS_int_definitions_list " ")
    #int, GNU conventions, sequential
    list(APPEND BLAS_lib_list "-lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm")
    list(APPEND BLAS_cxx_flag_list "")
    list(APPEND BLAS_int_definitions_list " ")

    #int64_t, Intel conventions, sequential
    list(APPEND BLAS_lib_list "-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm")
    list(APPEND BLAS_cxx_flag_list "")
    list(APPEND BLAS_int_definitions_list "-DMKL_ILP64")
    #int64_t, GNU conventions, sequential
    list(APPEND BLAS_lib_list "-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core -lm")
    list(APPEND BLAS_cxx_flag_list "")
    list(APPEND BLAS_int_definitions_list "-DMKL_ILP64")
endif()

list(LENGTH BLAS_lib_list blas_list_len)
list(LENGTH BLAS_int_definitions_list blas_int_list_len)

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

#set(BLAS_INT_DEFINES "")
set(BLAS_DEFINES "")
set(LIB_DEFINES "")
set(BLAS_links "")
set(BLAS_int "")

set(i 0)
foreach (lib_name ${BLAS_names})
    set(j 0)
    foreach(fortran_name ${fortran_mangling_names})
        set(k 0)
        foreach(int_size_name ${BLAS_int_size_names})
            list(GET fortran_mangling ${j} fort_var)

            list(GET BLAS_lib_list ${i} lib_var)
            list(GET BLAS_cxx_flag_list ${i} cxx_flag)
            list(GET BLAS_int_definitions_list ${i} int_define_var)

            list(GET BLAS_int_size_defines ${k} int_size_var)
            if("${int_define_var}" STREQUAL " ")
                set(int_define_var "")
            endif()
            if("${cxx_flag}" STREQUAL " ")
                set(cxx_flag "")
            endif()
            if("${int_size_var}" STREQUAL " ")
                set(int_size_var "")
            endif()

            message ("  ${i},${j},${k} - Trying: ${fortran_name}, ${lib_name}, ${int_size_name}")

            try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
                LINK_LIBRARIES
                ${lib_var}
                ${cxx_flag}
                COMPILE_DEFINITIONS
                ${fort_var}
                ${int_define_var}
                ${int_size_var}
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
                message("  Integer size:  ${int_size_var}${ColourReset}")

                LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
                set(BLAS_DEFINES "HAVE_BLAS" CACHE INTERNAL "")
                #set(config_found "TRUE")

                #set(BLAS_links ${lib_var})
                set(BLAS_links "${lib_var}" CACHE INTERNAL "")
                set(BLAS_cxx_flags "${cxx_flag}" CACHE INTERNAL "")
                set(BLAS_int "${int_define_var}" CACHE INTERNAL "")
                set(BLAS_int_size "${int_size_var}" CACHE INTERNAL "")

                # Break out of MKL checks if we found a working config
                break()
            else()
                message("${Red}  No${ColourReset}")
            endif()

            set(run_res1 "")
            set(compile_res1 "")
            set(run_output1 "")

            math(EXPR k "${k}+1")
            #if(config_found STREQUAL "TRUE")
            #    break()
            #endif()
        endforeach()
        if(BLAS_DEFINES STREQUAL "HAVE_BLAS")
            break()
        endif()
        math(EXPR j "${j}+1")
        if(NOT (j LESS fort_list_len))
            break()
        endif()
    endforeach ()
    # Break out of MKL checks if we found a working config
    if(BLAS_DEFINES STREQUAL "HAVE_BLAS")
        break()
    endif()
    math(EXPR i "${i}+1")
endforeach ()

if(NOT BLAS_DEFINES STREQUAL "HAVE_BLAS")
    message("Checking other libraries")
    set(BLAS_lib_list "")
    set(BLAS_names "")

    if("${BLAS_LIBRARY}" STREQUAL "auto" OR "${BLAS_LIBRARY}" STREQUAL "AMD ACML")
        list(APPEND BLAS_lib_list "-lacml_mp")
        list(APPEND BLAS_names "AMD ACML threaded")
        list(APPEND BLAS_lib_list "-lacml")
        list(APPEND BLAS_names "AMD ACML sequential")
    endif()
    if("${BLAS_LIBRARY}" STREQUAL "auto" OR "${BLAS_LIBRARY}" STREQUAL "IBM ESSL")
        list(APPEND BLAS_lib_list "-lessl")
        list(APPEND BLAS_names "IBM ESSL")
    endif()
    if("${BLAS_LIBRARY}" STREQUAL "auto" OR "${BLAS_LIBRARY}" STREQUAL "OpenBLAS")
        list(APPEND BLAS_lib_list "-lopenblas")
        list(APPEND BLAS_names "OpenBLAS")
    endif()

    set(i 0)
    foreach (lib_name ${BLAS_names})
        set(j 0)
        foreach(fortran_name ${fortran_mangling_names})
            set(k 0)
            foreach(int_size_name ${BLAS_int_size_names})
                list(GET fortran_mangling ${j} fort_var)
                list(GET BLAS_lib_list ${i} lib_var)
                #list(GET BLAS_cxx_flag_list ${i} cxx_flag)
                set(cxx_flag "-fopenmp")
                list(GET BLAS_int_size_defines {k} int_size_var)

                message ("  ${i},${j} - Trying: ${fortran_name}, ${lib_name}")

                try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
                    SOURCES
                    ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
                    LINK_LIBRARIES
                    ${lib_var}
                    ${cxx_flag}
                    COMPILE_DEFINITIONS
                    ${fort_var}
                    ${int_size_var}
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
                    message("  CXX flags: ${cxx_flag}${ColourReset}")

                    LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
                    set(BLAS_DEFINES "HAVE_BLAS")
                    #set(config_found "TRUE")

                    set(BLAS_links ${lib_var})
                    set(BLAS_cxx_flags ${cxx_flag})

                    # Break out of BLAS library checks if we found a working config
                    break()
                else()
                    message("${Red}  No${ColourReset}")
                endif()

                set(run_res1 "")
                set(compile_res1 "")
                set(run_output1 "")

                if("${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
                    message("config found - k loop")
                    break()
                endif()

                math(EXPR k "${k}+1")
            endforeach()

            if("${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
                message("config found - j loop")
                break()
            endif()

            math(EXPR j "${j}+1")
            if(NOT (j LESS fort_list_len))
                break()
            endif()
        endforeach ()
        #if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        #    break()
        #endif()
        # Break out of BLAS library checks if we found a working config
        #if(config_found STREQUAL "TRUE")
        if("${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
            #message("${Red}  FAILED TO FIND BLAS CONFIG${ColourReset}")
            message("config found - i loop")
            break()
        endif()
        math(EXPR i "${i}+1")
    endforeach ()
endif()

if(NOT BLAS_DEFINES STREQUAL "HAVE_BLAS")
#if("${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
    message("Checking Apple Accelerate library")
    set(BLAS_lib_list "")
    set(BLAS_names "")

    list(APPEND BLAS_lib_list "-framework Accelerate")
    list(APPEND BLAS_names "Apple Accelerate")

    set(i 0)
    foreach (lib_name ${BLAS_names})
        set(j 0)
        foreach(fortran_name ${fortran_mangling_names})
            list(GET fortran_mangling ${j} fort_var)
            list(GET BLAS_lib_list ${i} lib_var)
            #list(GET BLAS_cxx_flag_list ${i} cxx_flag)
            set(cxx_flag "-fopenmp")
            #list(GET BLAS_int_definitions_list ${i} int_define_var)

            message ("  ${i},${j} - Trying: ${fortran_name}, ${lib_name}")

            try_run(run_res1 compile_res1 ${CMAKE_CURRENT_BINARY_DIR}
                SOURCES
                ${CMAKE_CURRENT_SOURCE_DIR}/config/blas.cc
                LINK_LIBRARIES
                ${lib_var}
                ${cxx_flag}
                "-I/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers"
                COMPILE_DEFINITIONS
                ${fort_var}
                #${int_define_var}
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
                message("  CXX flags: ${cxx_flag}${ColourReset}")

                LIST(GET fortran_mangling_clean ${j} FORTRAN_MANGLING_DEFINES)
                set(BLAS_DEFINES "HAVE_BLAS")
                #set(config_found "TRUE")

                set(BLAS_links ${lib_var})
                set(BLAS_cxx_flags ${cxx_flag})

                # Break out of BLAS library checks if we found a working config
                break()
            else()
                message("${Red}  No${ColourReset}")
            endif()

            set(run_res1 "")
            set(compile_res1 "")
            set(run_output1 "")

            math(EXPR j "${j}+1")
            if(NOT (j LESS fort_list_len))
                break()
            endif()
        endforeach ()
        #if (compile_res1 AND NOT ${run_res1} MATCHES "FAILED_TO_RUN")
        #    break()
        #endif()
        # Break out of BLAS library checks if we found a working config
        if("${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
            #message("${Red}  FAILED TO FIND BLAS CONFIG${ColourReset}")
            break()
        endif()
        math(EXPR i "${i}+1")
    endforeach ()
endif()

if(NOT "${BLAS_DEFINES}" STREQUAL "HAVE_BLAS")
    message("${Red}  FAILED TO FIND BLAS LIBRARY${ColourReset}")
    return()
endif()

message("1 - blas libraries: " ${BLAS_LIBRARIES})
set(BLAS_LIBRARIES ${BLAS_links})
message("2 - blas libraries: " ${BLAS_LIBRARIES})

#if(DEBUG)
    #message("lib defines: " ${LIB_DEFINES})
    message("blas defines: " ${BLAS_DEFINES})
    message("blas libraries: " ${BLAS_LIBRARIES})
    message("cxx flags: " ${BLAS_cxx_flags})
    message("mkl int defines: " ${BLAS_int})
    message("fortran mangling defines: " ${FORTRAN_MANGLING_DEFINES})
    #message("blas complex return: " ${BLAS_RETURN})
    message("config_found: " ${config_found})
#endif()

if(config_found STREQUAL "TRUE")
    set(blas_config_found "TRUE")
    message("FOUND BLAS CONFIG")
endif()
