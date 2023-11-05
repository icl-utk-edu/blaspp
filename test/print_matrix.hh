// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef PRINT_HH
#define PRINT_HH

#include <assert.h>
#include <complex>

// -----------------------------------------------------------------------------
template <typename T>
void print_matrix( int64_t m, int64_t n, T *A, int64_t lda,
                   const char* format="%9.4f" )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );
    char format2[32];
    snprintf( format2, sizeof(format2), " %s", format );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf( format2, A(i, j) );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
template <typename T>
void print_matrix( int64_t m, int64_t n, std::complex<T>* A, int64_t lda,
                   const char* format="%9.4f" )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );
    char format2[32];
    snprintf( format2, sizeof(format2), " %s + %si", format, format );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf( format2, real(A(i, j)), imag(A(i, j)) );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
template <typename T>
void print_vector( int64_t n, T *x, int64_t incx,
                   const char* format="%9.4f" )
{
    assert( n >= 0 );
    assert( incx != 0 );
    char format2[32];
    snprintf( format2, sizeof(format2), " %s", format );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        printf( format2, x[ix] );
        ix += incx;
    }
    printf( " ]';\n" );
}

// -----------------------------------------------------------------------------
template <typename T>
void print_vector( int64_t n, std::complex<T>* x, int64_t incx,
                   const char* format="%9.4f" )
{
    assert( n >= 0 );
    assert( incx != 0 );
    char format2[32];
    snprintf( format2, sizeof(format2), " %s + %si", format, format );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        printf( format2, real(x[ix]), imag(x[ix]) );
        ix += incx;
    }
    printf( " ]';\n" );
}

#endif        //  #ifndef PRINT_HH
