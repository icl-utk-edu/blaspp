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
// get device counts
device_blas_int get_device_count()
{
    device_blas_int dev_counts;

    #ifdef BLAS_HAVE_CUBLAS
        blas_cuda_call(
            cudaGetDeviceCount(&dev_counts) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: rocBLAS equivalent
    #endif

    return dev_counts;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device trans constant
device_trans_t device_trans_const(blas::Op trans)
{
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );

    device_trans_t trans_ = DevNoTrans;
    switch (trans) {
        case Op::NoTrans:   trans_ = DevNoTrans;   break;
        case Op::Trans:     trans_ = DevTrans;     break;
        case Op::ConjTrans: trans_ = DevConjTrans; break;
        default:;
    }
    return trans_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device diag constant
device_diag_t device_diag_const(blas::Diag diag)
{
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );

    device_diag_t diag_ = DevDiagUnit;
    switch (diag) {
        case Diag::Unit:    diag_ = DevDiagUnit;    break;
        case Diag::NonUnit: diag_ = DevDiagNonUnit; break;
        default:;
    }
    return diag_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device uplo constant
device_uplo_t device_uplo_const(blas::Uplo uplo)
{
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );

    device_uplo_t uplo_ = DevUploLower;
    switch (uplo) {
        case Uplo::Upper: uplo_ = DevUploUpper; break;
        case Uplo::Lower: uplo_ = DevUploLower; break;
        default:;
    }
    return uplo_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device side constant
device_side_t device_side_const(blas::Side side)
{
    blas_error_if( side != Side::Left &&
                   side != Side::Right );

    device_side_t side_ = DevSideLeft;
    switch (side) {
        case Side::Left:  side_ = DevSideLeft;  break;
        case Side::Right: side_ = DevSideRight; break;
        default:;
    }
    return side_;
}

// -----------------------------------------------------------------------------
>>>>>>> 0b6db5b... Add device count and memcpy 2d
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
