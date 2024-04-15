// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>
#include <string.h>

namespace blas {

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
    scalar_t* y, int64_t incy,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_swap_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dev_swap );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::swap( n_, x, incx_, y, incy_, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup swap
void swap(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy,
    blas::Queue& queue )
{
    impl::swap( n, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup swap
void swap(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy,
    blas::Queue& queue )
{
    impl::swap( n, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup swap
void swap(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue )
{
    impl::swap( n, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup swap
void swap(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue )
{
    impl::swap( n, x, incx, y, incy, queue );
}

}  // namespace blas
