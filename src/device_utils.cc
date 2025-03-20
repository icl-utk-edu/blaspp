// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#if defined(BLAS_HAVE_CUBLAS)
    #include "cuda.h"   // CUDA_VERSION
#endif

#include "device_internal.hh"

namespace blas {

// -----------------------------------------------------------------------------
/// Set the current GPU device as needed by the accelerator/gpu.
/// (CUDA, ROCm only; no-op for SYCL.)
void internal_set_device( int device )
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaSetDevice( device ) );

    #elif defined( BLAS_HAVE_ROCBLAS )
        blas_dev_call(
            hipSetDevice( device ) );

    #elif defined( BLAS_HAVE_SYCL )
        // skip, no need to throw error since this is an internal function

    #else
        throw blas::Error( "unknown accelerator/gpu", __func__ );
    #endif
}

// -----------------------------------------------------------------------------
/// @return number of GPU devices. If BLAS++ is not compiled with GPU
/// support or any error occurs querying for GPUs (e.g., no GPUs found
/// or GPU driver not installed), returns 0. Does not throw an error.
int get_device_count()
{
    device_blas_int dev_count = 0;

    #ifdef BLAS_HAVE_CUBLAS
        auto err = cudaGetDeviceCount( &dev_count );
        if (err != cudaSuccess)
            dev_count = 0;

    #elif defined( BLAS_HAVE_ROCBLAS )
        auto err = hipGetDeviceCount( &dev_count );
        if (err != hipSuccess)
            dev_count = 0;

    #elif defined( BLAS_HAVE_SYCL )
        dev_count = DeviceList::size();
    #endif

    return dev_count;
}

// -----------------------------------------------------------------------------
/// Free a device memory space, allocated with device_malloc,
/// on the queue's device.
void device_free( void* ptr, blas::Queue &queue )
{
    #ifdef BLAS_HAVE_CUBLAS
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            cudaFree( ptr ) );

    #elif defined( BLAS_HAVE_ROCBLAS )
        blas::internal_set_device( queue.device() );
        blas_dev_call(
            hipFree( ptr ) );

    #elif defined( BLAS_HAVE_SYCL )
        blas_dev_call(
            sycl::free( ptr, queue.stream() ) );
    #endif
}

// -----------------------------------------------------------------------------
/// Free a pinned host memory space, allocated with host_malloc_pinned.
void host_free_pinned( void* ptr, blas::Queue &queue )
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaFreeHost( ptr ) );

    #elif defined( BLAS_HAVE_ROCBLAS )
        blas_dev_call(
            hipHostFree( ptr ) );

    #elif defined( BLAS_HAVE_SYCL )
        blas_dev_call(
            sycl::free( ptr, queue.stream() ) );

    #else
        throw blas::Error( "device BLAS not available", __func__ );
    #endif
}

// -----------------------------------------------------------------------------
/// Check whether the given pointer is allocated on device.
bool is_devptr( const void* A, blas::Queue &queue )
{
    #ifdef BLAS_HAVE_CUBLAS
        cudaError_t err;
        cudaPointerAttributes attr;
        err = cudaPointerGetAttributes( &attr, const_cast<void*>( A ) );
        if (! err) {
            #if CUDA_VERSION >= 11000
                return attr.type == cudaMemoryTypeDevice;
            #else
                return attr.memoryType == cudaMemoryTypeDevice;
            #endif
        }
        cudaGetLastError();

    #elif defined( BLAS_HAVE_ROCBLAS )
        hipError_t err;
        hipPointerAttribute_t attr;
        err = hipPointerGetAttributes( &attr, const_cast<void*>( A ) );
        if (! err) {
            #if HIP_VERSION >= 60000
                return attr.type == hipMemoryTypeDevice;
            #else
                return attr.memoryType == hipMemoryTypeDevice;
            #endif
        }
        err = hipGetLastError();

    #elif defined( BLAS_HAVE_SYCL )
            sycl::queue syclq = queue.stream();
            auto ptr_type = sycl::get_pointer_type( A, syclq.get_context() );
            return ptr_type == sycl::usm::alloc::device;
    #endif

    return false;
}

}  // namespace blas
