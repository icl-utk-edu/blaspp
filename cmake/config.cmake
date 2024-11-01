# Copyright (c) 2017-2024, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#-------------------------------------------------------------------------------
# Tests whether using `std::atomic` requires linking with `-latomic`
# for 64-bit values, which is the case on some 32-bit systems.
# Sets variable `libatomic_required`.
#
function( check_libatomic )
    message( STATUS "Checking whether std::atomic requires libatomic" )
    set( libatomic_required false )

    try_compile(
        link_result ${CMAKE_CURRENT_BINARY_DIR}
        SOURCES
            "${CMAKE_CURRENT_SOURCE_DIR}/config/std_atomic.cc"
        OUTPUT_VARIABLE
            link_output
    )
    debug_try_compile( "std_atomic.cc" "${link_result}" "${link_output}" )

    set( label "   std::atomic links without -latomic" )
    pad_string( "${label}" 50 label )
    if (link_result)
        message( "${label} ${blue} yes${plain}" )
    else()
        message( "${label} ${red} no${plain}" )

        try_compile(
            link_result ${CMAKE_CURRENT_BINARY_DIR}
            SOURCES
                "${CMAKE_CURRENT_SOURCE_DIR}/config/std_atomic.cc"
            LINK_LIBRARIES
                "-latomic"
            OUTPUT_VARIABLE
                link_output
        )
        debug_try_compile( "std_atomic.cc" "${link_result}" "${link_output}" )

        set( label "   std::atomic requires -latomic" )
        pad_string( "${label}" 50 label )
        if (link_result)
            #target_link_libraries( ${tgt} PUBLIC "-latomic" )
            message( "${label} ${blue} yes${plain}" )
            set( libatomic_required true )
        else()
            message( "${label} ${red} failed; cannot compile libatomic test${plain}" )
        endif()
    endif()
endfunction()
