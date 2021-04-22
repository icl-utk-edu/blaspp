// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef UTIL_H
#define UTIL_H

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

#endif // UTIL_H
