// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Function definitions moved from cblas_wrappers.hh for ESSL compatability.

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

// Circa 2022-12-22, there was a conflict in BLAS_*rot[g] when including
// both fortran.h and cblas.h (via cblas_wrappers.hh) on macOS Ventura.
// Can't replicate it now, and we need lapack_uplo_const() from
// cblas_wrappers.hh
#include "cblas_wrappers.hh"

#include <complex>

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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
        (blas_complex_float*) x, &incx_,
        (blas_complex_float*) y, &incy_,
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
        (blas_complex_double*) x, &incx_,
        (blas_complex_double*) y, &incy_,
        &c,
        (blas_complex_double*) &s );
}

//------------------------------------------------------------------------------
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<float> alpha,
    std::complex<float> const* A, int lda,
    std::complex<float> const* x, int incx,
    std::complex<float> beta,
    std::complex<float>* yref, int incy )
{
    blas_int n_    = blas_int( n );
    blas_int incx_ = blas_int( incx );
    blas_int incy_ = blas_int( incy );
    blas_int lda_  = blas_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    BLAS_csymv(
        &uplo_, &n_,
        (blas_complex_float*) &alpha,
        (blas_complex_float*) A, &lda_,
        (blas_complex_float*) x, &incx_,
        (blas_complex_float*) &beta,
        (blas_complex_float*) yref, &incy_
    );
}

//------------------------------------------------------------------------------
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<double> alpha,
    std::complex<double> const* A, int lda,
    std::complex<double> const* x, int incx,
    std::complex<double> beta,
    std::complex<double>* yref, int incy )
{
    blas_int n_    = blas_int( n );
    blas_int incx_ = blas_int( incx );
    blas_int incy_ = blas_int( incy );
    blas_int lda_  = blas_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    BLAS_zsymv(
        &uplo_, &n_,
        (blas_complex_double*) &alpha,
        (blas_complex_double*) A, &lda_,
        (blas_complex_double*) x, &incx_,
        (blas_complex_double*) &beta,
        (blas_complex_double*) yref, &incy_
    );
}

//------------------------------------------------------------------------------
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    blas_int n_    = blas_int( n );
    blas_int incx_ = blas_int( incx );
    blas_int lda_  = blas_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    BLAS_csyr(
        &uplo_, &n_,
        (blas_complex_float*) &alpha,
        (blas_complex_float*) x, &incx_,
        (blas_complex_float*) A, &lda_
    );
}

//------------------------------------------------------------------------------
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    blas_int n_    = blas_int( n );
    blas_int incx_ = blas_int( incx );
    blas_int lda_  = blas_int( lda );
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    BLAS_zsyr(
        &uplo_, &n_,
        (blas_complex_double*) &alpha,
        (blas_complex_double*) x, &incx_,
        (blas_complex_double*) A, &lda_
    );
}
