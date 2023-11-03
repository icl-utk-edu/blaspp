# Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# CMake script for TestSweeper library.

cmake_minimum_required( VERSION 3.15 )
# 3.1  target_compile_features
# 3.8  target_compile_features( cxx_std_17 )
# 3.14 install( LIBRARY DESTINATION lib ) default
# 3.15 $<$COMPILE_LANG_AND_ID  # optional
# 3.15 message DEBUG

project(
    testsweeper
    VERSION 2023.08.25
    LANGUAGES CXX
)

include( CheckCXXCompilerFlag )

# When built as a sub-project, add a namespace to make targets unique,
# e.g., `make example` becomes `make testsweeper_example`.
if (CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set( testsweeper_is_project true )
    set( testsweeper_ "" )
else()
    set( testsweeper_is_project false )
    set( testsweeper_ "testsweeper_" )
endif()

#-------------------------------------------------------------------------------
# Options
option( BUILD_SHARED_LIBS "Build shared libraries" true )
option( build_tests "Build test suite" "${testsweeper_is_project}" )
option( color "Use ANSI color output" true )
option( use_openmp "Use OpenMP, if available" true )
option( testsweeper_install "Add install target" "${testsweeper_is_project}" )

# Default prefix=/opt/slate
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT
    AND testsweeper_is_project)

    set( CMAKE_INSTALL_PREFIX "/opt/slate"
         CACHE PATH
         "Install path prefix, prepended onto install directories."
         FORCE
    )
    message( STATUS "Setting CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
    # Append the new CMAKE_INSTALL_PREFIX, since CMake appended the old value.
    list( APPEND CMAKE_SYSTEM_PREFIX_PATH ${CMAKE_INSTALL_PREFIX} )
else()
    message( STATUS "Using CMAKE_INSTALL_PREFIX = ${CMAKE_INSTALL_PREFIX}" )
endif()

# Provide menu of options. (Why doesn't CMake do this?)
set_property( CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
              None Debug Release RelWithDebInfo MinSizeRel )

message( DEBUG "Settings:
CMAKE_INSTALL_PREFIX   = ${CMAKE_INSTALL_PREFIX}
CMAKE_BUILD_TYPE       = ${CMAKE_BUILD_TYPE}
BUILD_SHARED_LIBS      = ${BUILD_SHARED_LIBS}
build_tests            = ${build_tests}
color                  = ${color}
use_openmp             = ${use_openmp}
testsweeper_install    = ${testsweeper_install}
testsweeper_is_project = ${testsweeper_is_project}
testsweeper_           = ${testsweeper_}
" )

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling TestSweeper with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
# Build library.
add_library(
    testsweeper
    testsweeper.cc
    version.cc
)

# Include directory.
# During build it's {source}; after install it's {prefix}/include.
target_include_directories(
    testsweeper
    PUBLIC
        "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>"
        "$<INSTALL_INTERFACE:include>"
)

# OpenMP support.
if (NOT use_openmp)
    message( STATUS "User has requested to NOT use OpenMP" )
else()
    find_package( OpenMP )
    if (OpenMP_CXX_FOUND)
        target_link_libraries( testsweeper PUBLIC OpenMP::OpenMP_CXX )
    endif()
endif()

# Get git commit id.
if (EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process( COMMAND git rev-parse --short HEAD
                     WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                     OUTPUT_VARIABLE testsweeper_id )
    string( STRIP "${testsweeper_id}" testsweeper_id )
    message( STATUS "testsweeper_id = ${testsweeper_id}" )
    target_compile_definitions(
        testsweeper PRIVATE TESTSWEEPER_ID="${testsweeper_id}" )
endif()

# Use and export -std=c++11; don't allow -std=gnu++11 extensions.
target_compile_features( testsweeper PUBLIC cxx_std_11 )
set_target_properties( testsweeper PROPERTIES CXX_EXTENSIONS false )

if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.15)
    # Conditionally add -Wall. See CMake tutorial.
    set( gcc_like_cxx "$<COMPILE_LANG_AND_ID:CXX,ARMClang,AppleClang,Clang,GNU>" )
    target_compile_options(
        testsweeper PRIVATE "$<${gcc_like_cxx}:$<BUILD_INTERFACE:-Wall>>" )
endif()

# Handle NaN correctly with Intel icpx.
check_cxx_compiler_flag( "-fp-model=precise" fp_model )
if (fp_model)
    target_compile_options( testsweeper PUBLIC -fp-model=precise )
endif()

# Avoid annoying icpx warning: 'linker' input unused
check_cxx_compiler_flag( "-Wno-unused-command-line-argument" warn_unused )
if (warn_unused)
    target_compile_options( testsweeper PUBLIC -Wno-unused-command-line-argument )
endif()

if (NOT color)
    target_compile_definitions( testsweeper PUBLIC "NO_COLOR" )
endif()

#-------------------------------------------------------------------------------
if (build_tests)
    add_subdirectory( test )
endif()

#-------------------------------------------------------------------------------
# Add 'make lib' target.
if (testsweeper_is_project)
    add_custom_target( lib DEPENDS testsweeper )
endif()

#-------------------------------------------------------------------------------
# Install rules.
# When TestSweeper is used as a sub-project,
# the parent (e.g., BLAS++) may not want to install it.
if (testsweeper_install)
    # GNU Filesystem Conventions
    include( GNUInstallDirs )
    if (WIN32)
        set( install_configdir "testsweeper" )
    else()
        set( install_configdir "${CMAKE_INSTALL_LIBDIR}/cmake/testsweeper" )
    endif()

    # Install library and add to <package>Targets.cmake
    install(
        TARGETS testsweeper
        EXPORT testsweeperTargets
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

    # Install header files
    install(
        FILES "testsweeper.hh"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    )

    # Install <package>Targets.cmake
    install(
        EXPORT testsweeperTargets
        DESTINATION "${install_configdir}"
    )

    # Also export <package>Targets.cmake in build directory
    export(
        EXPORT testsweeperTargets
        FILE "testsweeperTargets.cmake"
    )

    # Install <package>Config.cmake and <package>ConfigVersion.cmake,
    # to enable find_package( <package> ).
    include( CMakePackageConfigHelpers )
    configure_package_config_file(
        "testsweeperConfig.cmake.in"
        "testsweeperConfig.cmake"
        INSTALL_DESTINATION "${install_configdir}"
    )
    write_basic_package_version_file(
        "testsweeperConfigVersion.cmake"
        VERSION "${testsweeper_VERSION}"
        COMPATIBILITY AnyNewerVersion
    )
    install(
        FILES "${CMAKE_CURRENT_BINARY_DIR}/testsweeperConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/testsweeperConfigVersion.cmake"
        DESTINATION "${install_configdir}"
    )
endif()
