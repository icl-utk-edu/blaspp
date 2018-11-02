# Locate libtest for testing of BLAS++, LAPACK++, and SLATE

if(Libtest_FOUND)
    return()
endif()

find_path(Libtest_INCLUDE_DIR
    NAMES
        libtest.hh
    HINTS
        ${CMAKE_CURRENT_SOURCE_DIR}/../libtest
        ${CMAKE_CURRENT_SOURCE_DIR}/../../libtest
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../libtest
)

#message("libtest inc dir: ${Libtest_INCLUDE_DIR}")

find_library(Libtest_LIBRARY
    NAMES
        liblibtest.so liblibtest.a
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../libtest
        ${CMAKE_CURRENT_SOURCE_DIR}/../libtest/build
        ${CMAKE_CURRENT_SOURCE_DIR}/../../libtest/build
    NO_DEFAULT_PATH
)

if(Libtest_LIBRARY)
    set(Libtest_FOUND TRUE)
endif()

mark_as_advanced(Libtest_FOUND)

#message("libtest: ${Libtest_LIBRARY}")
#message("libtest found: ${Libtest_FOUND}")
#message("libtest dir: ${Libtest__INCLUDE_DIR}")

if(Libtest_FOUND)
    add_library(Libtest UNKNOWN IMPORTED)
    set_target_properties(Libtest
        PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${Libtest_INCLUDE_DIR}"
            IMPORTED_LOCATION
                "${Libtest_LIBRARY}"
    )
endif()