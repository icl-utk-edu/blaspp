// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DOTU_HH
#define BLAS_DOTU_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// @return unconjugated dot product, $x^T y$.
/// @see dot for conjugated version, $x^H y$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
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
/// @ingroup dotu

template< typename TX, typename TY >
scalar_type<TX, TY> dotu(
    int64_t n,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy )
{
    typedef scalar_type<TX, TY> scalar_t;

    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    scalar_t result = 0;
    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += x[i] * y[i];
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            result += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return result;
}

}  // namespace blas

#endif        //  #ifndef BLAS_DOTU_HH
