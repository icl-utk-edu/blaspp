# CXX compiler must match the one used to compiler BLAS++.
# Set it in your environment.

cmake_minimum_required( VERSION 3.8 )

project(
    blaspp_example
    LANGUAGES CXX
)

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
