// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup axpy_internal
///
template <typename scalar_t>
void axpy(
    int64_t n,
    scalar_t alpha,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue)
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::axpy( n_, alpha, x, incx_, y, incy_, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup axpy
void axpy(
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float*       y, int64_t incy,
    blas::Queue& queue)
{
    impl::axpy( n, alpha, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup axpy
void axpy(
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double*       y, int64_t incy,
    blas::Queue& queue)
{
    impl::axpy( n, alpha, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup axpy
void axpy(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>*       y, int64_t incy,
    blas::Queue& queue)
{
    impl::axpy( n, alpha, x, incx, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup axpy
void axpy(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>*       y, int64_t incy,
    blas::Queue& queue)
{
    impl::axpy( n, alpha, x, incx, y, incy, queue );
}

}  // namespace blas
