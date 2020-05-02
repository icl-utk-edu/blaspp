// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GEMV_HH
#define BLAS_GEMV_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// General matrix-vector multiply:
/// \[
///     y = \alpha op(A) x + \beta y,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// alpha and beta are scalars, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans:   $y = \alpha A   x + \beta y$,
///     - Op::Trans:     $y = \alpha A^T x + \beta y$,
///     - Op::ConjTrans: $y = \alpha A^H x + \beta y$.
///
/// @param[in] m
///     Number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, m) [RowMajor: lda >= max(1, n)].
///
/// @param[in] x
///     - If trans = NoTrans:
///       the n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///     - Otherwise:
///       the m-element vector x, in an array of length (m-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     - If trans = NoTrans:
///       the m-element vector y, in an array of length (m-1)*abs(incy) + 1.
///     - Otherwise:
///       the n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup gemv

template< typename TA, typename TX, typename TY >
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
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
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (m == 0 || n == 0 || (alpha == zero && beta == one))
        return;

    bool doconj = false;
    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => A & conj
        std::swap( m, n );
        if (trans == Op::NoTrans) {
            trans = Op::Trans;
        }
        else {
            if (trans == Op::ConjTrans) {
                doconj = true;
            }
            trans = Op::NoTrans;
        }
    }

    int64_t lenx = (trans == Op::NoTrans ? n : m);
    int64_t leny = (trans == Op::NoTrans ? m : n);
    int64_t kx = (incx > 0 ? 0 : (-lenx + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-leny + 1)*incy);

    // ----------
    // form y = beta*y
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int64_t i = 0; i < leny; ++i) {
                    y[i] = zero;
                }
            }
            else {
                for (int64_t i = 0; i < leny; ++i) {
                    y[i] *= beta;
                }
            }
        }
        else {
            int64_t iy = ky;
            if (beta == zero) {
                for (int64_t i = 0; i < leny; ++i) {
                    y[iy] = zero;
                    iy += incy;
                }
            }
            else {
                for (int64_t i = 0; i < leny; ++i) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
    }
    if (alpha == zero)
        return;

    // ----------
    if (trans == Op::NoTrans && ! doconj) {
        // form y += alpha * A * x
        int64_t jx = kx;
        if (incy == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                for (int64_t i = 0; i < m; ++i) {
                    y[i] += tmp * A(i, j);
                }
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                int64_t iy = ky;
                for (int64_t i = 0; i < m; ++i) {
                    y[iy] += tmp * A(i, j);
                    iy += incy;
                }
            }
        }
    }
    else if (trans == Op::NoTrans && doconj) {
        // form y += alpha * conj( A ) * x
        // this occurs for row-major A^H * x
        int64_t jx = kx;
        if (incy == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                for (int64_t i = 0; i < m; ++i) {
                    y[i] += tmp * conj(A(i, j));
                }
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                int64_t iy = ky;
                for (int64_t i = 0; i < m; ++i) {
                    y[iy] += tmp * conj(A(i, j));
                    iy += incy;
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // form y += alpha * A^T * x
        int64_t jy = ky;
        if (incx == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += A(i, j) * x[i];
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += A(i, j) * x[ix];
                    ix += incx;
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
    }
    else {
        // form y += alpha * A^H * x
        int64_t jy = ky;
        if (incx == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += conj(A(i, j)) * x[i];
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += conj(A(i, j)) * x[ix];
                    ix += incx;
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_GEMV_HH
