// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SCAL_HH
#define BLAS_SCAL_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Scale vector by constant, $x = \alpha x$.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup scal

template< typename T >
void scal(
    int64_t n,
    T alpha,
    T* x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }
    else {
        // non-unit stride
        for (int64_t i = 0; i < n; i += incx) {
            x[i] *= alpha;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_SCAL_HH
