// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_NRM2_HH
#define BLAS_NRM2_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// @return 2-norm of vector,
///     $|| x ||_2 = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}$.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic implementation does not currently scale to avoid over- or underflow.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup nrm2

template< typename T >
real_type<T>
nrm2(
    int64_t n,
    T const * x, int64_t incx )
{
    typedef real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: scale to avoid overflow & underflow
    real_t result = 0;
    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += real(x[i]) * real(x[i]) + imag(x[i]) * imag(x[i]);
        }
    }
    else {
        // non-unit stride
        int64_t ix = 0;
        for (int64_t i = 0; i < n; ++i) {
            result += real(x[ix]) * real(x[ix]) + imag(x[ix]) * imag(x[ix]);
            ix += incx;
        }
    }
    return std::sqrt( result );
}

}  // namespace blas

#endif        //  #ifndef BLAS_NRM2_HH
