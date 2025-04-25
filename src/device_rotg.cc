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
/// @ingroup rotg_internal
///
template <typename scalar_t>
void rotg(
    scalar_t* a,
    scalar_t* b,
    real_type<scalar_t>* c,
    scalar_t* s,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_rotg_type element;
        memset( &element, 0, sizeof( element ) );
        element = { 1 };
        counter::insert( element, counter::Id::dev_rotg );

        // This operation does not incur significant FLOPs, so no
        // need to call counter::inc_flop_count()
    #endif

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::rotg( a, b, c, s, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup rotg
void rotg(
    float *a,
    float *b,
    float *c,
    float *s,
    blas::Queue& queue )
{
    impl::rotg( a, b, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup rotg
void rotg(
    double *a,
    double *b,
    double *c,
    double *s,
    blas::Queue& queue )
{
    impl::rotg( a, b, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup rotg
void rotg(
    std::complex<float> *a,
    std::complex<float> *b,
    float *c,
    std::complex<float> *s,
    blas::Queue& queue )
{
    impl::rotg( a, b, c, s, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup rotg
void rotg(
    std::complex<double> *a,
    std::complex<double> *b,
    double *c,
    std::complex<double> *s,
    blas::Queue& queue )
{
    impl::rotg( a, b, c, s, queue );
}

}  // namespace blas