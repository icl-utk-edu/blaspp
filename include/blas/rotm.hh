// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
/// TODO: generic version not yet implemented.
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

template< typename TX, typename TY >
void rotm(
    int64_t n,
    TX *x, int64_t incx,
    TY *y, int64_t incy,
    blas::scalar_type<TX, TY> const param[5] )
{
    throw std::exception();  // not yet implemented

    // // check arguments
    // blas_error_if( n < 0 );
    // blas_error_if( incx == 0 );
    // blas_error_if( incy == 0 );
    //
    // if (incx == 1 && incy == 1) {
    //     // unit stride
    //     for (int64_t i = 0; i < n; ++i) {
    //         // TODO
    //     }
    // }
    // else {
    //     // non-unit stride
    //     int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    //     int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
    //     for (int64_t i = 0; i < n; ++i) {
    //         // TODO
    //         ix += incx;
    //         iy += incy;
    //     }
    // }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTM_HH
