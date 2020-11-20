// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_TYPES_HH
#define BLAS_DEVICE_TYPES_HH

#include "blas/defines.h"

#ifdef BLAS_HAVE_CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#elif defined(BLAS_HAVE_ROCBLAS)
    #include <hip/hip_runtime.h>
    #include <rocblas.h>
#endif

namespace blas {

// -----------------------------------------------------------------------------
// types

#ifdef BLAS_HAVE_CUBLAS
    typedef int                  device_blas_int;
    typedef cudaError_t          device_error_t;
    typedef cublasStatus_t       device_blas_status_t;
    typedef cublasHandle_t       device_blas_handle_t;
    typedef cublasOperation_t    device_trans_t;
    typedef cublasDiagType_t     device_diag_t;
    typedef cublasFillMode_t     device_uplo_t;
    typedef cublasSideMode_t     device_side_t;
    typedef cudaMemcpyKind       device_memcpy_t;

#elif defined(BLAS_HAVE_ROCBLAS)
    typedef int                  device_blas_int;
    typedef hipError_t           device_error_t;
    typedef rocblas_status       device_blas_status_t;
    typedef rocblas_handle       device_blas_handle_t;
    typedef rocblas_operation    device_trans_t;
    typedef rocblas_diagonal     device_diag_t;
    typedef rocblas_fill         device_uplo_t;
    typedef rocblas_side         device_side_t;
    typedef hipMemcpyKind        device_memcpy_t;

#else
    typedef int                  device_blas_int;
    typedef void*                device_blas_handle_t;
    enum device_error_t          {};
    enum device_blas_status_t    {};
    enum device_trans_t          {};
    enum device_diag_t           {};
    enum device_uplo_t           {};
    enum device_side_t           {};
    enum device_memcpy_t         {};

#endif

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_TYPES_HH
