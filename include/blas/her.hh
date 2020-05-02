// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HER_HH
#define BLAS_HER_HH

#include "blas/util.hh"
#include "blas/syr.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Hermitian matrix rank-1 update:
/// \[
///     A = \alpha x x^H + A,
/// \]
/// where alpha is a scalar, x is a vector,
/// and A is an n-by-n Hermitian matrix.
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
///     Imaginary parts of the diagonal elements need not be set,
///     are assumed to be zero on entry, and are set to zero on exit.
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @ingroup her

template< typename TA, typename TX >
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    blas::real_type<TA, TX> alpha,  // zher takes double alpha; use real
    TX const *x, int64_t incx,
    TA       *A, int64_t lda )
{
    typedef blas::scalar_type<TA, TX> scalar_t;
    typedef blas::real_type<TA, TX> real_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const real_t zero = 0;

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
                scalar_t tmp = alpha * conj( x[j] );
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i, j) += x[i] * tmp;
                }
                A(j, j) = real( A(j, j) ) + real( x[j] * tmp );
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( x[jx] );
                int64_t ix = kx;
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i, j) += x[ix] * tmp;
                    ix += incx;
                }
                A(j, j) = real( A(j, j) ) + real( x[jx] * tmp );
                jx += incx;
            }
        }
    }
    else {
        // lower triangle
        if (incx == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( x[j] );
                A(j, j) = real( A(j, j) ) + real( tmp * x[j] );
                for (int64_t i = j+1; i < n; ++i) {
                    A(i, j) += x[i] * tmp;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( x[jx] );
                A(j, j) = real( A(j, j) ) + real( tmp * x[jx] );
                int64_t ix = jx;
                for (int64_t i = j+1; i < n; ++i) {
                    ix += incx;
                    A(i, j) += x[ix] * tmp;
                }
                jx += incx;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER_HH
