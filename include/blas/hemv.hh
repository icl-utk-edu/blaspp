// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HEMV_HH
#define BLAS_HEMV_HH

#include "blas/util.hh"
#include "blas/symv.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Hermitian matrix-vector multiply:
/// \[
///     y = \alpha A x + \beta y,
/// \]
/// where alpha and beta are scalars, x and y are vectors,
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
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///     Imaginary parts of the diagonal elements need not be set,
///     and are assumed to be zero.
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup hemv

template< typename TA, typename TX, typename TY >
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TA const *A, int64_t lda,
    TX const *x, int64_t incx,
    blas::scalar_type<TA, TX, TY> beta,
    TY *y, int64_t incy )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (n == 0 || (alpha == zero && beta == one))
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);

    // form y = beta*y
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int64_t i = 0; i < n; ++i) {
                    y[i] = zero;
                }
            }
            else {
                for (int64_t i = 0; i < n; ++i) {
                    y[i] *= beta;
                }
            }
        }
        else {
            int64_t iy = ky;
            if (beta == zero) {
                for (int64_t i = 0; i < n; ++i) {
                    y[iy] = zero;
                    iy += incy;
                }
            }
            else {
                for (int64_t i = 0; i < n; ++i) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
    }
    if (alpha == zero)
        return;

    if (layout == Layout::ColMajor) {
        if (uplo == Uplo::Upper) {
            // A is stored in upper triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = 0; i < j; ++i) {
                        y[i] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = kx;
                    int64_t iy = ky;
                    for (int64_t i = 0; i < j; ++i) {
                        y[iy] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[ix];
                        ix += incx;
                        iy += incy;
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else if (uplo == Uplo::Lower) {
            // A is stored in lower triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = j+1; i < n; ++i) {
                        y[i] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = jx;
                    int64_t iy = jy;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        iy += incy;
                        y[iy] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[ix];
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }
    else {
        if (uplo == Uplo::Lower) {
            // A is stored in lower triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = 0; i < j; ++i) {
                        y[i] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = kx;
                    int64_t iy = ky;
                    for (int64_t i = 0; i < j; ++i) {
                        y[iy] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[ix];
                        ix += incx;
                        iy += incy;
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else if (uplo == Uplo::Upper) {
            // A is stored in upper triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = j+1; i < n; ++i) {
                        y[i] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = jx;
                    int64_t iy = jy;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        iy += incy;
                        y[iy] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[ix];
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMV_HH
