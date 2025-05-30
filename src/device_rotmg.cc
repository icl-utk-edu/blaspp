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
/// @ingroup rotmg_internal
///
template <typename scalar_t>
void rotmg(
    scalar_t* d1,
    scalar_t* d2,
    scalar_t* x1,
    scalar_t* y1,
    scalar_t* param,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_rotmg_type element;
        memset( &element, 0, sizeof( element ) );
        element = { 1 };
        counter::insert( element, counter::Id::dev_rotmg );

        // This operation does not incur significant FLOPs, so no
        // need to call counter::inc_flop_count()
    #endif

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::rotmg( d1, d2, x1, y1, param, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
// Not available for complex.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup rotmg
void rotmg(
    float* d1,
    float* d2,
    float* x1,
    float* y1,
    float* param,
    blas::Queue& queue )
{
    impl::rotmg( d1, d2, x1, y1, param, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup rotmg
void rotmg(
    double* d1,
    double* d2,
    double* x1,
    double* y1,
    double* param,
    blas::Queue& queue )
{
    impl::rotmg( d1, d2, x1, y1, param, queue );
}

} // namespace blas
