// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef __HIP_PLATFORM_HCC__
#define __HIP_PLATFORM_HCC__
#endif

#include <hip/hip_runtime.h>

// Headers moved in ROCm 5.2
#if HIP_VERSION >= 50200000
    #include <rocblas/rocblas.h>
#else
    #include <rocblas.h>
#endif

#include <stdexcept>
#include <cassert>
#include <cstdio>

//------------------------------------------------------------------------------
void error_check_( hipError_t err, const char* file, int line )
{
    if (err != hipSuccess) {
        printf( "HIP error %d: %s at %s:%d\n",
                err, hipGetErrorString(err), file, line );
        exit(1);
    }
}

//------------------------------------------------------------------------------
void error_check_( rocblas_status err, const char* file, int line )
{
    if (err != rocblas_status_success) {
        printf( "rocblas error %d: %s at %s:%d\n",
                err, rocblas_status_to_string(err), file, line );
        exit(1);
    }
}

#define error_check( err ) \
        error_check_( (err), __FILE__, __LINE__ )

//------------------------------------------------------------------------------
int main()
{
    double alpha = 2, beta = 3;
    int n = 2;
    double A[] = { 1, 2, 3, 4 };
    double B[] = { 5, 4, 3, 2 };
    double C[] = { 2, 3, 1, 0 };
    double D[] = { 40, 61, 21, 28 };

    hipError_t err = hipSetDevice( 0 );
    if (err != hipSuccess) {
        printf( "hipSetDevice failed: %s (%d).\n"
                "Cannot run on GPU; skipping test.\n",
                hipGetErrorString(err), err );
        return 0;
    }

    double *dA, *dB, *dC;
    error_check(
        hipMalloc( &dA, n*n*sizeof(double) ) );
    error_check(
        hipMalloc( &dB, n*n*sizeof(double) ) );
    error_check(
        hipMalloc( &dC, n*n*sizeof(double) ) );
    assert( dA != nullptr );
    assert( dB != nullptr );
    assert( dC != nullptr );

    // dA = A, dB = B, dC = c
    error_check(
        hipMemcpy( dA, A, n*n*sizeof(double), hipMemcpyDefault ) );
    error_check(
        hipMemcpy( dB, B, n*n*sizeof(double), hipMemcpyDefault ) );
    error_check(
        hipMemcpy( dC, C, n*n*sizeof(double), hipMemcpyDefault ) );

    // C = alpha A B + beta C
    rocblas_handle handle;
    error_check(
        rocblas_create_handle( &handle ) );
    error_check(
        rocblas_dgemm( handle, rocblas_operation_none, rocblas_operation_none,
                       n, n, n,
                       &alpha, dA, n, dB, n, &beta, dC, n ) );
    error_check(
        rocblas_destroy_handle( handle ) );

    // C = dC
    error_check(
        hipMemcpy( C, dC, n*n*sizeof(double), hipMemcpyDefault ) );

    error_check(
        hipFree( dA ) );
    error_check(
        hipFree( dB ) );
    error_check(
        hipFree( dC ) );

    // verify C == D
    double result = 0;
    for (int i = 0; i < n*n; ++i) {
        printf( "C[%d] = %.2f, D = %.2f\n", i, C[i], D[i] );
        result += std::abs( D[i] - C[i] );
    }
    bool okay = (result == 0);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
