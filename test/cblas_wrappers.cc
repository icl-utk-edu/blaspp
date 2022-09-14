// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Function definitions moved from cblas_wrappers.hh for ESSL compatability.

#include "blas/fortran.h"
#include "cblas_wrappers.hh"

#include <complex>

// -----------------------------------------------------------------------------
void
cblas_rotg(
    std::complex<float> *a, std::complex<float> *b,
    float *c, std::complex<float> *s )
{
    BLAS_crotg(
        (blas_complex_float*) a,
        (blas_complex_float*) b,
        c,
        (blas_complex_float*) s );
}

void
cblas_rotg(
    std::complex<double> *a, std::complex<double> *b,
    double *c, std::complex<double> *s )
{
    BLAS_zrotg(
        (blas_complex_double*) a,
        (blas_complex_double*) b,
        c,
        (blas_complex_double*) s );
}

// -----------------------------------------------------------------------------
void
cblas_rot(
    int n,
    std::complex<float> *x, int incx,
    std::complex<float> *y, int incy,
    float c, std::complex<float> s )
{
    BLAS_crot(
        &n,
        (blas_complex_float*) x,
        &incx,
        (blas_complex_float*) y,
        &incy,
        &c,
        (blas_complex_float*) &s );
}

void
cblas_rot(
    int n,
    std::complex<double> *x, int incx,
    std::complex<double> *y, int incy,
    double c, std::complex<double> s )
{
    BLAS_zrot(
        &n,
        (blas_complex_double*) x,
        &incx,
        (blas_complex_double*) y,
        &incy,
        &c,
        (blas_complex_double*) &s );
}
