// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <stdexcept>
#include <cassert>
#include <cstdio>

//------------------------------------------------------------------------------
void error_check_( cudaError_t err, const char* file, int line )
{
    if (err != cudaSuccess) {
        printf( "CUDA error %d: %s at %s:%d\n",
                err, cudaGetErrorString(err), file, line );
        exit(1);
    }
}

//------------------------------------------------------------------------------
void error_check_( cublasStatus_t err, const char* file, int line )
{
    if (err != CUBLAS_STATUS_SUCCESS) {
        printf( "cuBLAS error %d at %s:%d\n",
                err, file, line );
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

    cudaError_t err = cudaSetDevice( 0 );
    if (err != cudaSuccess) {
        printf( "cudaSetDevice failed: %s (%d).\n"
                "Cannot run on GPU; skipping test.\n",
                cudaGetErrorString(err), err );
        return 0;
    }

    double *dA, *dB, *dC;
    error_check(
        cudaMalloc( &dA, n*n*sizeof(double) ) );
    error_check(
        cudaMalloc( &dB, n*n*sizeof(double) ) );
    error_check(
        cudaMalloc( &dC, n*n*sizeof(double) ) );
    assert( dA != nullptr );
    assert( dB != nullptr );
    assert( dC != nullptr );

    // dA = A, dB = B, dC = c
    error_check(
        cudaMemcpy( dA, A, n*n*sizeof(double), cudaMemcpyDefault ) );
    error_check(
        cudaMemcpy( dB, B, n*n*sizeof(double), cudaMemcpyDefault ) );
    error_check(
        cudaMemcpy( dC, C, n*n*sizeof(double), cudaMemcpyDefault ) );

    // C = alpha A B + beta C
    cublasHandle_t handle;
    error_check(
        cublasCreate( &handle ) );
    error_check(
        cublasDgemm( handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n,
                     &alpha, dA, n, dB, n, &beta, dC, n ) );
    error_check(
        cublasDestroy( handle ) );

    // C = dC
    error_check(
        cudaMemcpy( C, dC, n*n*sizeof(double), cudaMemcpyDefault ) );

    error_check(
        cudaFree( dA ) );
    error_check(
        cudaFree( dB ) );
    error_check(
        cudaFree( dC ) );

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
