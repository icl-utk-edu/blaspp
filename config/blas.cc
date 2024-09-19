// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <complex>

#include "config.h"

//------------------------------------------------------------------------------
#define BLAS_sdot FORTRAN_NAME( sdot, SDOT )
#define BLAS_ddot FORTRAN_NAME( ddot, DDOT )

#ifdef BLAS_HAVE_ACCELERATE
//ACCELERATE_NEW_LAPACK
    #pragma message "include Accelerate.h"
    #include <Accelerate/Accelerate.h>
#else
    // result returned directly
    #ifdef __cplusplus
    extern "C"
    #endif
    double BLAS_ddot(
        const blas_int* n,
        const double* x, const blas_int* incx,
        const double* y, const blas_int* incy );
#endif

//------------------------------------------------------------------------------
int main()
{
    // If blas_int is 32-bit, but BLAS actually interprets it as 64-bit,
    // BLAS will see n = 0x500000005 and segfault.
    // If blas_int is 64-bit, BLAS can interpret it as 32-bit or 64-bit
    // to see n = 5 and pass.
    blas_int n[] = { 5, 5 }, ione = 1;
    double x[] = { 1, 2, 3, 4, 5 };
    double y[] = { 5, 4, 3, 2, 1 };
    for (int i = 0; i < n[0]; ++i) {
        printf( "x[ %d ] = %.1f; y[ %d ] = %.1f\n",
                i, x[ i ],
                i, y[ i ] );
    }

    double result = BLAS_ddot( n, x, &ione, y, &ione );
    printf( "result = %.1f; should be 35.0\n", result );
    bool okay = (result == 35);

    #ifdef ACCELERATE_NEW_LAPACK
        // To verify this is new Accelerate, check the return type of sdot.
        float sx[] = { 1, 2, 3, 4, 5 };
        float sy[] = { 5, 4, 3, 2, 1 };
        auto r = BLAS_sdot( n, sx, &ione, sy, &ione );
        static_assert( std::is_same< float, decltype( r ) >::value,
                       "With new Accelerate, sdot returns float"
                       " -> this is old Accelerate" );
    #endif

    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
