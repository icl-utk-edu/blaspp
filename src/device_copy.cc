// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup copy
void blas::copy(
    int64_t n,
    float const *dx, int64_t incdx,
    float *dy, int64_t incdy,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incdx == 0 );
    blas_error_if( incdy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_      = (device_blas_int) n;
    device_blas_int incdx_  = (device_blas_int) incdx;
    device_blas_int incdy_  = (device_blas_int) incdy;

    #ifndef BLAS_HAVE_ONEMKL
    blas::set_device( queue.device() );
    #endif

    device::scopy(
        queue,
        n_,
        dx, incdx_,
        dy, incdy_);
}
// -----------------------------------------------------------------------------
/// @ingroup copy
void blas::copy(
    int64_t n,
    double const *dx, int64_t incdx,
    double *dy, int64_t incdy,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incdx == 0 );
    blas_error_if( incdy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_      = (device_blas_int) n;
    device_blas_int incdx_  = (device_blas_int) incdx;
    device_blas_int incdy_  = (device_blas_int) incdy;

    #ifndef BLAS_HAVE_ONEMKL
    blas::set_device( queue.device() );
    #endif

    device::dcopy(
        queue,
        n_,
        dx, incdx_,
        dy, incdy_);
}
// -----------------------------------------------------------------------------
/// @ingroup copy
void blas::copy(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    std::complex<float> *dy, int64_t incdy,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incdx == 0 );
    blas_error_if( incdy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_      = (device_blas_int) n;
    device_blas_int incdx_  = (device_blas_int) incdx;
    device_blas_int incdy_  = (device_blas_int) incdy;

    #ifndef BLAS_HAVE_ONEMKL
    blas::set_device( queue.device() );
    #endif

    device::ccopy(
        queue,
        n_,
        dx, incdx_,
        dy, incdy_);
}
// -----------------------------------------------------------------------------
/// @ingroup copy
void blas::copy(
    int64_t n,
    std::complex<double> const *dx, int64_t incdx,
    std::complex<double> *dy, int64_t incdy,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incdx == 0 );
    blas_error_if( incdy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdy > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_      = (device_blas_int) n;
    device_blas_int incdx_  = (device_blas_int) incdx;
    device_blas_int incdy_  = (device_blas_int) incdy;

    #ifndef BLAS_HAVE_ONEMKL
    blas::set_device( queue.device() );
    #endif

    device::zcopy(
        queue,
        n_,
        dx, incdx_,
        dy, incdy_);
}
