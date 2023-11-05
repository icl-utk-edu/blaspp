// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROT_HH
#define BLAS_ROT_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Apply plane rotation:
/// \[
///       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}
///     = \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
///       \begin{bmatrix} x^T   \\ y^T    \end{bmatrix}.
/// \]
///
/// @see rotg to generate the rotation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in, out] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param[in] c
///     Cosine of rotation; real.
///
/// @param[in] s
///     Sine of rotation; complex.
///
/// @ingroup rot

template <typename TX, typename TY>
void rot(
    int64_t n,
    TX *x, int64_t incx,
    TY *y, int64_t incy,
    blas::real_type<TX, TY>   c,
    blas::scalar_type<TX, TY> s )
{
    typedef scalar_type<TX, TY> scalar_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    scalar_t zero( 0 );

    // quick return
    if (n == 0 || (c == 1 && s == zero))
        return;

    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            scalar_t stmp = c*x[i] + s*y[i];
            y[i] = c*y[i] - conj(s)*x[i];
            x[i] = stmp;
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            scalar_t stmp = c*x[ix] + s*y[iy];
            y[iy] = c*y[iy] - conj(s)*x[ix];
            x[ix] = stmp;
            ix += incx;
            iy += incy;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROT_HH
