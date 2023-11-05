// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <sycl/detail/cl.h>
#include <sycl.hpp>
#include <oneapi/mkl.hpp>

#include <stdexcept>
#include <cassert>
#include <cstdio>

//------------------------------------------------------------------------------
int main()
{
  try {
    double alpha = 2, beta = 3;
    int n = 2;
    double A[] = { 1, 2, 3, 4 };
    double B[] = { 5, 4, 3, 2 };
    double C[] = { 2, 3, 1, 0 };
    double D[] = { 40, 61, 21, 28 };

    // enumerate devices
    std::vector< sycl::device > devices;
    auto platforms = sycl::platform::get_platforms();
    for (auto& platform : platforms) {
        auto all_devices = platform.get_devices();
        for (auto& device : all_devices) {
            if (device.is_gpu()) {
                devices.push_back( device );
            }
        }
    }
    if (devices.size() == 0) {
        printf( "no sycl GPU devices\n" );
        return -1;
    }

    sycl::queue queue( devices[0] );

    double *dA, *dB, *dC;
    dA = (double*) sycl::malloc_shared( n*n*sizeof(double), queue );
    dB = (double*) sycl::malloc_shared( n*n*sizeof(double), queue );
    dC = (double*) sycl::malloc_shared( n*n*sizeof(double), queue );

    // dA = A, dB = B, dC = c
    queue.memcpy( dA, A, n*n*sizeof(double) );
    queue.memcpy( dB, B, n*n*sizeof(double) );
    queue.memcpy( dC, C, n*n*sizeof(double) );

    // C = alpha A B + beta C
    oneapi::mkl::blas::gemm(
        queue,
        oneapi::mkl::transpose::N, oneapi::mkl::transpose::N,
        n, n, n,
        alpha, dA, n, dB, n, beta, dC, n );

    // C = dC
    queue.memcpy( dC, C, n*n*sizeof(double) );

    sycl::free( dA, queue );
    sycl::free( dB, queue );
    sycl::free( dC, queue );

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
  catch (...) {
      printf( "caught error\n" );
      return -2;
  }
}
