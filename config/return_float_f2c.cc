// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

//------------------------------------------------------------------------------
#define BLAS_sdot FORTRAN_NAME( sdot, SDOT )

#ifdef ACCELERATE_NEW_LAPACK
    // New Accelerate API (>= macOS 13.3) does not use f2c convention.
    // Since new Accelerate requires using their prototypes in their header,
    // it's not possible to test using a custom prototype as below.
    #error "Accelerate's new API (>= macOS 13.3) does not use f2c convention."
#else
    // returns `double` instead of `float`, per f2c convention.
    #ifdef __cplusplus
    extern "C"
    #endif
    double BLAS_sdot( const blas_int* n,
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

    float result = BLAS_sdot( &n, x, &ione, y, &ione );
    printf( "result = %.1f; should be 35.0\n", result );

    bool okay = (result == 35);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
