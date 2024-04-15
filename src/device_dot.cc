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
/// Conjugated x^H y version.
/// @ingroup dot_internal
///
template <typename scalar_t>
void dot(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy,
    scalar_t* result,
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
        counter::dev_dot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dev_dot );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

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
            scalar_t* dev_work = (scalar_t*)queue.work();
            internal::dot( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::dot( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::dot( n_, x, incx_, y, incy_, result, queue );
    #endif
#endif
}

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// Unconjugated x^T y version.
/// @ingroup dotu_internal
///
template <typename scalar_t>
void dotu(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy,
    scalar_t* result,
    blas::Queue& queue )
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
    #if defined( BLAS_HAVE_SYCL )
        sycl::queue syclq = queue.stream();
        // check how the result scalar was allocated
        auto result_ptr_type = sycl::get_pointer_type( result, syclq.get_context() );
        // if result was outside SYCL/USM memory allocation, use device workspace
        if (result_ptr_type == sycl::usm::alloc::unknown) {
            // use preallocated device workspace (resizing if needed)
            queue.work_ensure_size< char >( sizeof(scalar_t) );  // syncs if needed
            scalar_t* dev_work = (scalar_t*)queue.work();
            internal::dotu( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::dotu( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::dotu( n_, x, incx_, y, incy_, result, queue );
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
/// @ingroup dot
void dot(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float* result,
    blas::Queue& queue)
{
    impl::dot( n, x, incx, y, incy, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup dot
void dot(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double* result,
    blas::Queue& queue)
{
    impl::dot( n, x, incx, y, incy, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup dot
void dot(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>* result,
    blas::Queue& queue)
{
    impl::dot( n, x, incx, y, incy, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup dot
void dot(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>* result,
    blas::Queue& queue)
{
    impl::dot( n, x, incx, y, incy, result, queue );
}

//==============================================================================
// Unconjugated x y^T versions.

//------------------------------------------------------------------------------
/// GPU device, float, unconjugated x^T y version.
/// @ingroup dotu
void dotu(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float* result,
    blas::Queue& queue)
{
    dot( n, x, incx, y, incy, result, queue );
}

// -----------------------------------------------------------------------------
/// GPU device, double, unconjugated x^T y version.
/// @ingroup dotu
void dotu(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double* result,
    blas::Queue& queue)
{
    dot( n, x, incx, y, incy, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float>, unconjugated x^T y version.
/// @ingroup dotu
void dotu(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>* result,
    blas::Queue& queue)
{
    impl::dotu( n, x, incx, y, incy, result, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double>, unconjugated x^T y version.
/// @ingroup dotu
void dotu(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>* result,
    blas::Queue& queue)
{
    impl::dotu( n, x, incx, y, incy, result, queue );
}

} // namespace blas
