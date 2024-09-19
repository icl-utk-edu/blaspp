// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <type_traits>

#include "config.h"

//------------------------------------------------------------------------------
#define BLAS_sdot FORTRAN_NAME( sdot, SDOT )

#ifdef BLAS_HAVE_ACCELERATE
//ACCELERATE_NEW_LAPACK
    #pragma message "include Accelerate.h"
    #include <Accelerate/Accelerate.h>
#else
    // returns `float` as usual.
    #ifdef __cplusplus
    extern "C"
    #endif
    float  BLAS_sdot( const blas_int* n,
                      const float* x, const blas_int* incx,
                      const float* y, const blas_int* incy );
#endif

//------------------------------------------------------------------------------
int main()
{
    blas_int n = 5, ione = 1;
    float x[] = { 1, 2, 3, 4, 5 };
    float y[] = { 5, 4, 3, 2, 1 };
    for (int i = 0; i < n; ++i) {
        printf( "x[ %d ] = %.1f; y[ %d ] = %.1f\n",
                i, x[ i ],
                i, y[ i ] );
    }

    auto r = BLAS_sdot( &n, x, &ione, y, &ione );
    if (! std::is_same<float, decltype(r)>::value) {
        printf( "is_same failed\n" );
    }

    float result = BLAS_sdot( &n, x, &ione, y, &ione );
    printf( "result = %.1f; should be 35.0\n", result );

    bool okay = (result == 35);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
