// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_TYPES_HH
#define BLAS_DEVICE_TYPES_HH

#ifdef BLASPP_WITH_CUBLAS
    #include <cuda_runtime.h>
    #include <cublas_v2.h>
#elif defined(HAVE_ROCBLAS)
    // TODO
#endif

namespace blas {

// -----------------------------------------------------------------------------
// types

#ifdef BLASPP_WITH_CUBLAS
    typedef    int                  device_blas_int;
    typedef    cudaError_t          device_error_t;
    typedef    cublasStatus_t       device_blas_status_t;
    typedef    cublasHandle_t       device_blas_handle_t;
    typedef    cublasOperation_t    device_trans_t;
    typedef    cublasDiagType_t     device_diag_t;
    typedef    cublasFillMode_t     device_uplo_t;
    typedef    cublasSideMode_t     device_side_t;
#elif defined(HAVE_ROCBLAS)
    // TODO: add rocBLAS types and constants
#endif

}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_TYPES_HH
