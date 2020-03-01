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
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // swap m <=> n, x <=> y
        BLAS_sger( &n_, &m_, &alpha, y, &incy_, x, &incx_, A, &lda_ );
    }
    else {
        BLAS_sger( &m_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // swap m <=> n, x <=> y
        BLAS_dger( &n_, &m_, &alpha, y, &incy_, x, &incx_, A, &lda_ );
    }
    else {
        BLAS_dger( &m_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // conjugate y (in y2)
        std::complex<float> *y2 = new std::complex<float>[n];
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y2[i] = conj( y[iy] );
            iy += incy;
        }
        incy_ = 1;

        // swap m <=> n, x <=> y, call geru
        BLAS_cgeru( &n_, &m_,
                    (blas_complex_float*) &alpha,
                    (blas_complex_float*) y2, &incy_,
                    (blas_complex_float*) x, &incx_,
                    (blas_complex_float*) A, &lda_ );

        delete[] y2;
    }
    else {
        BLAS_cgerc( &m_, &n_,
                    (blas_complex_float*) &alpha,
                    (blas_complex_float*) x, &incx_,
                    (blas_complex_float*) y, &incy_,
                    (blas_complex_float*) A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // conjugate y (in y2)
        std::complex<double> *y2 = new std::complex<double>[n];
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y2[i] = conj( y[iy] );
            iy += incy;
        }
        incy_ = 1;

        // swap m <=> n, x <=> y, call geru
        BLAS_zgeru( &n_, &m_,
                    (blas_complex_double*) &alpha,
                    (blas_complex_double*) y2, &incy_,
                    (blas_complex_double*) x, &incx_,
                    (blas_complex_double*) A, &lda_ );

        delete[] y2;
    }
    else {
        BLAS_zgerc( &m_, &n_,
                    (blas_complex_double*) &alpha,
                    (blas_complex_double*) x, &incx_,
                    (blas_complex_double*) y, &incy_,
                    (blas_complex_double*) A, &lda_ );
    }
}

}  // namespace blas
