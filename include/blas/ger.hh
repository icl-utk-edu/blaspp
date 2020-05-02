// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GER_HH
#define BLAS_GER_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// General matrix rank-1 update:
/// \[
///     A = \alpha x y^H + A,
/// \]
/// where alpha is a scalar, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] m
///     Number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The m-element vector x, in an array of length (m-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param[in, out] A
///     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1, n)].
///
/// @ingroup ger

template< typename TA, typename TX, typename TY >
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy,
    TA *A, int64_t lda )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;

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

    // quick return
    if (m == 0 || n == 0 || alpha == zero)
        return;

    if (layout == Layout::ColMajor) {
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if y[j] is zero, for consistent NAN handling
                scalar_t tmp = alpha * conj( y[j] );
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[i] * tmp;
                }
            }
        }
        else if (incx == 1) {
            // x unit stride, y non-unit stride
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( y[jy] );
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[i] * tmp;
                }
                jy += incy;
            }
        }
        else {
            // x and y non-unit stride
            int64_t kx = (incx > 0 ? 0 : (-m + 1)*incx);
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( y[jy] );
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    A(i, j) += x[ix] * tmp;
                    ix += incx;
                }
                jy += incy;
            }
        }
    }
    else {
        // RowMajor
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t i = 0; i < m; ++i) {
                // note: NOT skipping if x[i] is zero, for consistent NAN handling
                scalar_t tmp = alpha * x[i];
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * conj( y[j] );
                }
            }
        }
        else if (incy == 1) {
            // x non-unit stride, y unit stride
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                scalar_t tmp = alpha * x[ix];
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * conj( y[j] );
                }
                ix += incx;
            }
        }
        else {
            // x and y non-unit stride
            int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                scalar_t tmp = alpha * x[ix];
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    A(j, i) += tmp * conj( y[jy] );
                    jy += incy;
                }
                ix += incx;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_GER_HH
