// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <complex>

#include "config.h"

//------------------------------------------------------------------------------
#define BLAS_zdotc FORTRAN_NAME( zdotc, ZDOTC )

// result returned as *hidden argument*
#ifdef __cplusplus
extern "C"
#endif
void BLAS_zdotc(
    std::complex<double>* result,
    const blas_int* n,
    const std::complex<double>* x, const blas_int* incx,
    const std::complex<double>* y, const blas_int* incy );

//------------------------------------------------------------------------------
int main()
{
    blas_int n = 5, ione = 1;
    std::complex<double> x[] = { 1, 2, 3, 4, 5 };
    std::complex<double> y[] = { 5, 4, 3, 2, 1 };
    for (int i = 0; i < n; ++i) {
        printf( "x[ %d ] = %.1f + %.1fi; y[ %d ] = %.1f + %.1fi\n",
                i, real( x[ i ] ), imag( x[ i ] ),
                i, real( y[ i ] ), imag( y[ i ] ) );
    }

    std::complex<double> result;
    BLAS_zdotc( &result, &n, x, &ione, y, &ione );
    printf( "result = %.1f + %.1fi; should be 35.0 + 0.0i\n",
            real( result ), imag( result ) );

    bool okay = (real(result) == 35);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
