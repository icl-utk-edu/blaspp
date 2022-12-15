// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
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
/// @ingroup nrm2_internal
///
template <typename scalar_t>
void nrm2(
    int64_t n,
    scalar_t const* x, int64_t incx,
    real_type<scalar_t>* result,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::nrm2( n_, x, incx_, result, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// Unlike CPU version, here `result` is an output parameter,
/// to store the result when the asynchronous execution completes.
/// @ingroup nrm2
void nrm2(
    int64_t n,
    float const* x, int64_t incx,
    float* result,
    blas::Queue& queue )
{
    impl::nrm2( n, x, incx, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup nrm2
void nrm2(
    int64_t n,
    double const* x, int64_t incx,
    double* result,
    blas::Queue& queue )
{
    impl::nrm2( n, x, incx, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup nrm2
void nrm2(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    float* result,
    blas::Queue& queue )
{
    impl::nrm2( n, x, incx, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup nrm2
void nrm2(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    double* result,
    blas::Queue& queue )
{
    impl::nrm2( n, x, incx, result, queue );
}

}  // namespace blas
