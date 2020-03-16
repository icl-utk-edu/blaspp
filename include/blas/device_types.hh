// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_TYPES_HH
#define BLAS_DEVICE_TYPES_HH

#ifdef BLASPP_WITH_CUBLAS
// -----------------------------------------------------------------------------
// types
typedef    int                  device_blas_int;
typedef    cudaError_t          device_error_t;
typedef    cublasStatus_t       device_blas_status_t;
typedef    cublasHandle_t       device_blas_handle_t;
typedef    cublasOperation_t    device_trans_t;
typedef    cublasDiagType_t     device_diag_t;
typedef    cublasFillMode_t     device_uplo_t;
typedef    cublasSideMode_t     device_side_t;

// constants -- defined as needed.
// Only needed are those shared between cublas and rocblas
// Constants that exist on only one side should be explicitly handled
#define    DevSuccess        cudaSuccess
#define    DevBlasSuccess    CUBLAS_STATUS_SUCCESS

// trans
#define    DevNoTrans        CUBLAS_OP_N
#define    DevTrans          CUBLAS_OP_T
#define    DevConjTrans      CUBLAS_OP_C

// diag
#define    DevDiagUnit       CUBLAS_DIAG_UNIT
#define    DevDiagNonUnit    CUBLAS_DIAG_NON_UNIT

// uplo
#define    DevUploUpper      CUBLAS_FILL_MODE_UPPER
#define    DevUploLower      CUBLAS_FILL_MODE_LOWER

// side
#define    DevSideLeft       CUBLAS_SIDE_LEFT
#define    DevSideRight      CUBLAS_SIDE_RIGHT

#elif defined(HAVE_ROCBLAS)
// -----------------------------------------------------------------------------
// TODO: add rocBLAS types and constants
#endif

#endif        //  #ifndef BLAS_DEVICE_TYPES_HH
