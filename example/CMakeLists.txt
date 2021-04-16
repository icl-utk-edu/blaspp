# CXX compiler must match the one used to compiler BLAS++.
# Set it in your environment.

cmake_minimum_required( VERSION 3.8 )

project(
    blaspp_example
    LANGUAGES CXX
)

#-------------------------------------------------------------------------------
# Enforce out-of-source build
string( TOLOWER "${CMAKE_CURRENT_SOURCE_DIR}" source_dir )
string( TOLOWER "${CMAKE_CURRENT_BINARY_DIR}" binary_dir )
if ("${source_dir}" STREQUAL "${binary_dir}")
    message( FATAL_ERROR
    "Compiling with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_CURRENT_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()

#-------------------------------------------------------------------------------
find_package( blaspp REQUIRED )

#--------------------
add_executable(
    example_gemm
    example_gemm.cc
)
target_link_libraries(
    example_gemm
    blaspp
)

#--------------------
add_executable(
    example_util
    example_util.cc
)
target_link_libraries(
    example_util
    blaspp
)

#-------------------------------------------------------------------------------
# CTest
enable_testing()
add_test( NAME example_gemm COMMAND ./example_gemm )
add_test( NAME example_util COMMAND ./example_util )
