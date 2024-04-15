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

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_nrm2_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dev_nrm2 );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    #if defined( BLAS_HAVE_SYCL )
        sycl::queue syclq = queue.stream();
        // check how the result scalar was allocated
        auto result_ptr_type = sycl::get_pointer_type( result, syclq.get_context() );
        // if result was outside SYCL/USM memory allocation, use device workspace
        if (result_ptr_type == sycl::usm::alloc::unknown) {
            // use preallocated device workspace (resizing if needed)
            queue.work_ensure_size< char >( sizeof(scalar_t) );  // syncs if needed
            real_type<scalar_t>* dev_work = (real_type<scalar_t>*)queue.work();
            internal::nrm2( n_, x, incx_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::nrm2( n_, x, incx_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::nrm2( n_, x, incx_, result, queue );
    #endif
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
