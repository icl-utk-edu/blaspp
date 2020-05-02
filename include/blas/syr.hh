// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYR_HH
#define BLAS_SYR_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Symmetric matrix rank-1 update:
/// \[
///     A = \alpha x x^T + A,
/// \]
/// where alpha is a scalar, x is a vector,
/// and A is an n-by-n symmetric matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed from symmetry.
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in, out] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup syr

template< typename TA, typename TX >
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    blas::scalar_type<TA, TX> alpha,
    TX const *x, int64_t incx,
    TA       *A, int64_t lda )
{
    printf( "template syr implementation\n" );

    typedef blas::scalar_type<TA, TX> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( lda < n );

    // quick return
    if (n == 0 || alpha == zero)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    if (uplo == Uplo::Upper) {
        if (incx == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                scalar_t tmp = alpha * x[j];
                for (int64_t i = 0; i <= j; ++i) {
                    A(i, j) += x[i] * tmp;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[jx];
                int64_t ix = kx;
                for (int64_t i = 0; i <= j; ++i) {
                    A(i, j) += x[ix] * tmp;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }
    else {
        // lower triangle
        if (incx == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[j];
                for (int64_t i = j; i < n; ++i) {
                    A(i, j) += x[i] * tmp;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[jx];
                int64_t ix = jx;
                for (int64_t i = j; i < n; ++i) {
                    A(i, j) += x[ix] * tmp;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYR_HH
