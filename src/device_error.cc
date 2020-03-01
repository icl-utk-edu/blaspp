// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

// -----------------------------------------------------------------------------
// return true on runtime errors
bool blas::is_device_error(device_error_t error)
{
    return (error != DevSuccess);
}

// -----------------------------------------------------------------------------
// return true on blas errors
bool blas::is_device_error(device_blas_status_t status)
{
    return (status != DevBlasSuccess);
}

// -----------------------------------------------------------------------------
// return string of runtime error
const char* blas::device_error_string(device_error_t error)
{
    #ifdef BLASPP_WITH_CUBLAS
    return cudaGetErrorString( error );
    #elif defined(HAVE_ROCBLAS)
    // TODO: return error string for rocblas
    #endif
}

// -----------------------------------------------------------------------------
// return string of blas error
const char* blas::device_error_string(device_blas_status_t status)
{
    switch (status) {
    #ifdef BLASPP_WITH_CUBLAS
        case CUBLAS_STATUS_SUCCESS:
            return "device blas: success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "device blas: not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "device blas: out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "device blas: invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "device blas: architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "device blas: memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "device blas: execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "device blas: internal error";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "device blas: functionality not supported";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "device blas: license error";
    #elif defined(HAVE_ROCBLAS)
    // TODO: return error string for rocblas
    #endif
        default:
            return "unknown device blas error code";
    }
}
