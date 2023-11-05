// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTM_HH
#define BLAS_ROTM_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Apply modified (fast) plane rotation, H:
/// \[
///       \begin{bmatrix} x^T \\ y^T \end{bmatrix}
///     = H
///       \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
/// \]
///
/// @see rotmg to generate the rotation, and for fuller description.
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
/// @param[in] param
///     Array of length 5 giving parameters of modified plane rotation.
///
/// @ingroup rotm

template <typename TX, typename TY>
void rotm(
    int64_t n,
    TX *x, int64_t incx,
    TY *y, int64_t incy,
    blas::scalar_type<TX, TY> const param[5] )
{
    typedef scalar_type<TX, TY> scalar_t;

    // check arguments
    blas_error_if( n < 0 ); // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (n == 0 || param[0] == -2)
        return;

    if (incx == 1 && incy == 1) {
        // unit stride
        if (param[0] == -1) {
            const scalar_t& h11 = param[1];
            const scalar_t& h21 = param[2];
            const scalar_t& h12 = param[3];
            const scalar_t& h22 = param[4];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = h11*x[i] + h12*y[i];
                y[i] = h22*y[i] + h21*x[i];
                x[i] = stmp;
            }
        }
        else if (param[0] == 1) {
            const scalar_t& h11 = param[1];
            const scalar_t& h22 = param[4];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = h11*x[i] + y[i];
                y[i] = h22*y[i] - x[i];
                x[i] = stmp;
            }
        }
        else if (param[0] == 0) {
            const scalar_t& h21 = param[2];
            const scalar_t& h12 = param[3];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = x[i] + h12*y[i];
                y[i] = y[i] + h21*x[i];
                x[i] = stmp;
            }
        }
        else {
            throw Error("Invalid param[1] in blas::rotm");
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        if (param[0] == -1) {
            const scalar_t& h11 = param[1];
            const scalar_t& h21 = param[2];
            const scalar_t& h12 = param[3];
            const scalar_t& h22 = param[4];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = h11*x[ix] + h12*y[iy];
                y[iy] = h22*y[iy] + h21*x[ix];
                x[ix] = stmp;
                ix += incx;
                iy += incy;
            }
        }
        else if (param[0] == 1) {
            const scalar_t& h11 = param[1];
            const scalar_t& h22 = param[4];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = h11*x[ix] + y[iy];
                y[iy] = h22*y[iy] - x[ix];
                x[ix] = stmp;
                ix += incx;
                iy += incy;
            }
        }
        else if (param[0] == 0) {
            const scalar_t& h21 = param[2];
            const scalar_t& h12 = param[3];
            for (int64_t i = 0; i < n; ++i) {
                scalar_t stmp = x[ix] + h12*y[iy];
                y[iy] = y[iy] + h21*x[ix];
                x[ix] = stmp;
                ix += incx;
                iy += incy;
            }
        }
        else {
            throw Error("Invalid param[1] in blas::rotm");
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTM_HH
