#-------------------------------------------------------------------------------
set( tester "${testsweeper_}tester" )
add_executable(
    ${tester}
    test.cc
    test_sort.cc
)

# C++11 is inherited from testsweeper, but disabling extensions is not.
set_target_properties( ${tester} PROPERTIES CXX_EXTENSIONS false )

target_link_libraries(
    ${tester}
    testsweeper
)

# Copy run_tests script and reference output to build directory.
add_custom_command(
    TARGET ${tester} POST_BUILD
    COMMAND
        cp -a ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py
              ${CMAKE_CURRENT_SOURCE_DIR}/ref
              ${CMAKE_CURRENT_BINARY_DIR}/
)

if (testsweeper_is_project)
    add_custom_target(
        "check"
        COMMAND
            python3 run_tests.py
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    )
endif()
