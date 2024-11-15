// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

//------------------------------------------------------------------------------
/// @ingroup iamax
void blas::iamax(
    int64_t n,
    float const* dx, int64_t incdx,
    device_info_int* result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        // blas_error_if( result > std::numeric_limits<device_blas_int>::max() );
    }

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    // device_info_int* result_ = (device_info_int*) result;

    #ifndef BLAS_HAVE_ONEMKL
    blas::internal_set_device( queue.device() );
    #endif

    internal::isamax( n_, dx, incdx_, result, queue );
}

//------------------------------------------------------------------------------
/// @ingroup iamax
void blas::iamax(
    int64_t n,
    double const* dx, int64_t incdx,
    device_info_int* result,
    blas::Queue& queue)
{
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        // blas_error_if( result > std::numeric_limits<device_blas_int>::max() );
    }

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    // device_info_int* result_ = ( device_info_int* ) result;

    #ifndef BLAS_HAVE_ONEMKL
    blas::internal_set_device( queue.device() );
    #endif

    internal::idamax( n_, dx, incdx_, result, queue );
}

//------------------------------------------------------------------------------
/// @ingroup iamax
void blas::iamax(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    device_info_int* result,
    blas::Queue& queue)
{
        // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        // blas_error_if( result > std::numeric_limits<device_blas_int>::max() );
    }

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    // device_info_int* result_ = (device_info_int*) result;

    #ifndef BLAS_HAVE_ONEMKL
    blas::internal_set_device( queue.device() );
    #endif

    internal::icamax( n_, dx, incdx_, result, queue );
}

//------------------------------------------------------------------------------
/// @ingroup iamax
void blas::iamax(
    int64_t n,
    std::complex<double> const* dx, int64_t incdx,
    device_info_int* result,
    blas::Queue& queue)
{
        // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
        // blas_error_if( result > std::numeric_limits<device_blas_int>::max() );
    }

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    // device_info_int* result_ = (device_info_int*) result;

    #ifndef BLAS_HAVE_ONEMKL
    blas::internal_set_device( queue.device() );
    #endif

    internal::izamax( n_, dx, incdx_, result, queue );
}