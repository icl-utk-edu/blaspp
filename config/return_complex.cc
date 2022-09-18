// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <complex>

#include "config.h"

#define BLAS_zdotc FORTRAN_NAME( zdotc, ZDOTC )

// result return directly
#ifdef __cplusplus
extern "C"
#endif
std::complex<double> BLAS_zdotc(
    const blas_int* n,
    const std::complex<double>* x, const blas_int* incx,
    const std::complex<double>* y, const blas_int* incy );

int main()
{
    blas_int n = 5, ione = 1;
    std::complex<double> x[] = { 1, 2, 3, 4, 5 };
    std::complex<double> y[] = { 5, 4, 3, 2, 1 };
    std::complex<double> result = BLAS_zdotc( &n, x, &ione, y, &ione );
    bool okay = (real(result) == 35);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
