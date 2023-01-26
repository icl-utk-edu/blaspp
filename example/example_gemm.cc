// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <blas.hh>

#include <vector>
#include <stdio.h>

#include "util.hh"

//------------------------------------------------------------------------------
template <typename T>
void test_gemm( int m, int n, int k )
{
    print_func();

    int lda = m;
    int ldb = n;
    int ldc = m;
    std::vector<T> A( lda*k, 1.0 );  // m-by-k
    std::vector<T> B( ldb*n, 2.0 );  // k-by-n
    std::vector<T> C( ldc*n, 3.0 );  // m-by-n

    // ... fill in application data into A, B, C ...

    // C = -1.0*A*B + 1.0*C
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, n, k,
                -1.0, A.data(), lda,
                      B.data(), ldb,
                 1.0, C.data(), ldc );
}

//------------------------------------------------------------------------------
template <typename T>
void test_device_gemm( int m, int n, int k )
{
    print_func();
    if (blas::get_device_count() == 0) {
        printf( "no GPU devices\n" );
    }
    else {
        int lda = m;
        int ldb = n;
        int ldc = m;
        std::vector<T> A( lda*k, 1.0 );  // m-by-k
        std::vector<T> B( ldb*n, 2.0 );  // k-by-n
        std::vector<T> C( ldc*n, 3.0 );  // m-by-n

        // ... fill in application data into A, B, C ...

        int device = 0;
        int batch_size = 1000;  // todo: use default batch_size
        blas::Queue queue( device, batch_size );

        T *dA = blas::device_malloc<T>( lda*k, queue );  // m-by-k
        T *dB = blas::device_malloc<T>( ldb*n, queue );  // k-by-n
        T *dC = blas::device_malloc<T>( ldc*n, queue );  // m-by-n

        blas::device_copy_matrix(
            m, k,
            A.data(), lda,      // src
            dA, lda, queue );   // dst

        blas::device_copy_matrix(
            k, n,
            B.data(), ldb,      // src
            dB, ldb, queue );   // dst

        blas::device_copy_matrix(
            m, n,
            C.data(), ldc,      // src
            dC, ldc, queue );   // dst

        // C = -1.0*A*B + 1.0*C
        blas::gemm(
            blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
            m, n, k,
            -1.0, dA, lda,
                  dB, ldb,
             1.0, dC, ldc,
            queue );

        blas::device_copy_matrix(
            m, n,
            dC, ldc,                 // src
            C.data(), ldc, queue );  // dst

        queue.sync();

        blas::device_free( dA, queue );  dA = nullptr;
        blas::device_free( dB, queue );  dB = nullptr;
        blas::device_free( dC, queue );  dC = nullptr;
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m = 100, n = 200, k = 50;
    test_gemm< float  >( m, n, k );
    test_gemm< double >( m, n, k );
    test_gemm< std::complex<float>  >( m, n, k );
    test_gemm< std::complex<double> >( m, n, k );

    test_device_gemm< float  >( m, n, k );
    test_device_gemm< double >( m, n, k );
    test_device_gemm< std::complex<float>  >( m, n, k );
    test_device_gemm< std::complex<double> >( m, n, k );

    return 0;
}
