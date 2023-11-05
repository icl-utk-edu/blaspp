// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <stdexcept>

//------------------------------------------------------------------------------
void print_func_( const char* func )
{
    printf( "\n%s\n", func );
}

#ifdef __GNUC__
    #define print_func() print_func_( __PRETTY_FUNCTION__ )
#else
    #define print_func() print_func_( __func__ )
#endif

//------------------------------------------------------------------------------
// Parse command line options:
// s = single,         sets types[ 0 ]
// d = double,         sets types[ 1 ]
// c = complex,        sets types[ 2 ]
// z = double-complex, sets types[ 3 ]
// If no options, sets all types to true.
// Throws error for unknown options.
void parse_args( int argc, char** argv, bool types[ 4 ] )
{
    if (argc == 1) {
        types[ 0 ] = types[ 1 ] = types[ 2 ] = types[ 3 ] = true;
    }
    else {
        types[ 0 ] = types[ 1 ] = types[ 2 ] = types[ 3 ] = false;
    }
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[ i ];
        if (arg == "s")
            types[ 0 ] = true;
        else if (arg == "d")
            types[ 1 ] = true;
        else if (arg == "c")
            types[ 2 ] = true;
        else if (arg == "z")
            types[ 3 ] = true;
        else {
            throw std::runtime_error(
                "unknown option: \"" + arg + "\"\n"
                + "Usage: " + argv[ 0 ] + " [s] [d] [c] [z]\n"
                + "for single, double, complex, double-complex.\n" );
        }
    }
}

#endif // UTIL_H
