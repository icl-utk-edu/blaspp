// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Function definitions moved from cblas_wrappers.hh for ESSL compatability.

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

// Including a variant of <cblas.h> can cause conflicts in BLAS_*rot[g]
// Fortran prototypes, e.g., on macOS Ventura. So we define these without
// including their prototypes.
//#include "cblas_wrappers.hh"

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
    blas_int n_ = n;
    blas_int incx_ = incx;
    blas_int incy_ = incy;
    BLAS_crot(
        &n_,
        (blas_complex_float*) x,
        &incx_,
        (blas_complex_float*) y,
        &incy_,
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
    blas_int n_ = n;
    blas_int incx_ = incx;
    blas_int incy_ = incy;
    BLAS_zrot(
        &n_,
        (blas_complex_double*) x,
        &incx_,
        (blas_complex_double*) y,
        &incy_,
        &c,
        (blas_complex_double*) &s );
}
