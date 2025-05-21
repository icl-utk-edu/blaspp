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
/// TX is data [x, y]
/// TS is for sine, which can be real (zdrot) or complex (zrot)
/// cosine is always real
/// @ingroup rot_internal
///
template <typename TX, typename TS>
void rot(
    int64_t n,
    TX* x, int64_t incx,
    TX* y, int64_t incy,
    const real_type<TX> c,
    const TS s,
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
        counter::dev_rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dev_rot );

        double gflops = 1e9 * blas::Gflop< float >::rot( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::rot( n_, x, incx_, y, incy_, c, s, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup rot
void rot(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy,
    const float c,
    const float s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup rot
void rot(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy,
    const double c,
    const double s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup rot
/// real cosine, real sine
/// This variant, with real sine, is used in the SVD or EVD where a real
/// Givens rotation, that eliminates entries in the real bi- or
/// tridiagonal matrix, is applied to complex singular or eigen vectors.
///
void rot(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    const float c,
    const float s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup rot
/// real cosine, real sine
void rot(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    const double c,
    const double s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup rot
/// real cosine, complex sine
/// This variant, with complex sine, is used to eliminate entries in a
/// complex matrix.
///
void rot(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    const float c,
    const std::complex<float> s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup rot
/// real cosine, complex sine
void rot(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    const double c,
    const std::complex<double> s,
    blas::Queue& queue )
{
    impl::rot( n, x, incx, y, incy, c, s, queue );
}

}  // namespace blas
