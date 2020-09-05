# CXX compiler must match the one used to compiler BLAS++.
# Set it in your environment.

cmake_minimum_required( VERSION 3.8 )

project(
    blaspp_example
    LANGUAGES CXX
)

add_executable(
    example_gemm
    example_gemm.cc
)

find_package( blaspp )

target_link_libraries(
    example_gemm
    blaspp
)
