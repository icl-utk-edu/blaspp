// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>


// -----------------------------------------------------------------------------
namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup iamax
void iamax(
    int64_t n,
    float const *dx, int64_t incdx,
    int64_t *result,
    blas::Queue& queue)
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;

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
            internal::iamax( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::iamax( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::isamax( n_, x, incx_, y, incy_, result, queue );
    #endif
#endif
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
void iamax(
    int64_t n,
    double const *dx, int64_t incdx,
    int64_t *result,
    blas::Queue& queue)
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;

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
            internal::iamax( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::iamax( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::idamax( n_, x, incx_, y, incy_, result, queue );
    #endif
#endif
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
void iamax(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    int64_t *result,
    blas::Queue& queue)
{
    #ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;

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
            internal::iamax( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::iamax( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::icamax( n_, x, incx_, y, incy_, result, queue );
    #endif
#endif
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
void iamax(
    int64_t n,
    std::complex<double> const *dx, int64_t incdx,
    int64_t *result,
    blas::Queue& queue)
{
    #ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( n < 0 );       // standard BLAS returns, doesn't fail
    blas_error_if( incdx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n     > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( incdx > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_     = (device_blas_int) n;
    device_blas_int incdx_ = (device_blas_int) incdx;

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
            internal::iamax( n_, x, incx_, y, incy_, dev_work, queue );
            blas::device_memcpy( result, dev_work, 1, queue );
        }
        else {
            internal::iamax( n_, x, incx_, y, incy_, result, queue );
        }
    #else // other devices (CUDA/HIP)
        internal::izamax( n_, x, incx_, y, incy_, result, queue );
    #endif
#endif
}
} // namespace blas