// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup scal
void scal(
    int64_t n,
    float alpha,
    float *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    BLAS_sscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
void scal(
    int64_t n,
    double alpha,
    double *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    BLAS_dscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    BLAS_cscal( &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    BLAS_zscal( &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx_ );
}

}  // namespace blas
