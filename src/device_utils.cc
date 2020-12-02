// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#include "device_internal.hh"

namespace blas {

// -----------------------------------------------------------------------------
// set device
void set_device(blas::Device device)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaSetDevice((device_blas_int)device) );
    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipSetDevice((device_blas_int)device) );
    #endif
}

// -----------------------------------------------------------------------------
// get current device
void get_device(blas::Device *device)
{
    device_blas_int dev;

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaGetDevice(&dev) );
    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipGetDevice(&dev) );
    #endif

    (*device) = (blas::Device)dev;
}

// -----------------------------------------------------------------------------
// @return number of GPU devices
device_blas_int get_device_count()
{
    device_blas_int dev_counts;

    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaGetDeviceCount(&dev_counts) );
    #elif defined(BLAS_HAVE_ROCBLAS)
         blas_dev_call(
            hipGetDeviceCount(&dev_counts) );
    #endif

    return dev_counts;
}

// -----------------------------------------------------------------------------
/// free a device pointer
void device_free(void* ptr)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaFree( ptr ) );
    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipFree( ptr ) );
    #endif
}

// -----------------------------------------------------------------------------
/// free a pinned memory space
void device_free_pinned(void* ptr)
{
    #ifdef BLAS_HAVE_CUBLAS
        blas_dev_call(
            cudaFreeHost( ptr ) );
    #elif defined(BLAS_HAVE_ROCBLAS)
        blas_dev_call(
            hipHostFree( ptr ) );
    #endif
}

}  // namespace blas
