// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas_internal.hh"
#include "blas/counter.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup swap_internal
inline void swap(
    blas_int n,
    float* x, blas_int incx,
    float* y, blas_int incy )
{
    BLAS_sswap( &n, x, &incx, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup swap_internal
inline void swap(
    blas_int n,
    double* x, blas_int incx,
    double* y, blas_int incy )
{
    BLAS_dswap( &n, x, &incx, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup swap_internal
inline void swap(
    blas_int n,
    std::complex<float>* x, blas_int incx,
    std::complex<float>* y, blas_int incy )
{
    BLAS_cswap( &n,
                (blas_complex_float*) x, &incx,
                (blas_complex_float*) y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup swap_internal
inline void swap(
    blas_int n,
    std::complex<double>* x, blas_int incx,
    std::complex<double>* y, blas_int incy )
{
    BLAS_zswap( &n,
                (blas_complex_double*) x, &incx,
                (blas_complex_double*) y, &incy );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup swap_internal
///
template <typename scalar_t>
void swap(
    int64_t n,
    scalar_t* x, int64_t incx,
    scalar_t* y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::swap_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::swap );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    // call low-level wrapper
    internal::swap( n_, x, incx_, y, incy_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup swap
void swap(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy )
{
    impl::swap( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup swap
void swap(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy )
{
    impl::swap( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup swap
void swap(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy )
{
    impl::swap( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup swap
void swap(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy )
{
    impl::swap( n, x, incx, y, incy );
}

}  // namespace blas
