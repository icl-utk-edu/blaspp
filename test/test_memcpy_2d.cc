// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "print_matrix.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"

// -----------------------------------------------------------------------------
template <typename T>
void test_memcpy_2d_work( Params& params, bool run )
{
    using namespace testsweeper;
    using real_t = blas::real_type<T>;

    // get & mark input values
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t device  = params.device();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.time2();
    params.time3();
    params.time4();
    params.gbytes();
    params.gbytes2();
    params.gbytes3();
    params.gbytes4();
    params.ref_time();
    params.ref_gbytes();

    params.time    .name( "h2d (sec)" );
    params.time2   .name( "d2d (sec)" );
    params.time3   .name( "d2h (sec)" );
    params.time4   .name( "h2h (sec)" );
    params.ref_time.name( "ref (sec)" );

    params.gbytes    .name( "h2d GB/s" );
    params.gbytes2   .name( "d2d GB/s" );
    params.gbytes3   .name( "d2h GB/s" );
    params.gbytes4   .name( "h2h GB/s" );
    params.ref_gbytes.name( "ref GB/s" );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    enum class Method {
        memcpy_2d,
        copy_matrix,
        set_matrix,   // tests both set_matrix and get_matrix
    };

    Method method;
    if (params.routine == "memcpy_2d") {
        method = Method::memcpy_2d;
    }
    else if (params.routine == "copy_matrix") {
        method = Method::copy_matrix;
    }
    else if (params.routine == "set_matrix") {
        method = Method::set_matrix;
    }
    else {
        throw blas::Error( "unknown method" );
    }

    // setup
    blas::Queue queue( device );

    // Allocate extra to verify copy doesn't overrun buffer.
    int64_t ld = roundup( m, align );
    int64_t extra = 2;
    int64_t size = ld*(n + extra);
    T* a_host = blas::host_malloc_pinned<T>( size, queue );
    T* b_host = blas::host_malloc_pinned<T>( size, queue );
    T* c_host = blas::host_malloc_pinned<T>( size, queue );
    T* d_host = blas::host_malloc_pinned<T>( size, queue );

    // device specifics
    T* b_dev = blas::device_malloc<T>( size, queue );
    T* c_dev = blas::device_malloc<T>( size, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size, a_host );
    lapack_larnv( idist, iseed, size, b_host );
    lapack_larnv( idist, iseed, size, c_host );
    lapack_larnv( idist, iseed, size, d_host );

    // test error exits
    assert_throw( blas::device_memcpy_2d( c_dev, m-1, b_dev, m,    m,  n, queue ), blas::Error );
    assert_throw( blas::device_memcpy_2d( c_dev, m,   b_dev, m-1,  m,  n, queue ), blas::Error );
    assert_throw( blas::device_memcpy_2d( c_dev, m,   b_dev, m,   -1,  n, queue ), blas::Error );
    assert_throw( blas::device_memcpy_2d( c_dev, m,   b_dev, m,    m, -1, queue ), blas::Error );

    assert_throw( blas::device_copy_matrix( -1,  n, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m, -1, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m-1, c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m,   c_dev, m-1, queue ), blas::Error );

    assert_throw( blas::device_copy_matrix( -1,  n, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m, -1, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m-1, c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m,   c_dev, m-1, queue ), blas::Error );

    assert_throw( blas::device_copy_matrix( -1,  n, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m, -1, b_dev, m,   c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m-1, c_dev, m,   queue ), blas::Error );
    assert_throw( blas::device_copy_matrix(  m,  n, b_dev, m,   c_dev, m-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "m=%5lld, n=%5lld, ld=%2lld\n",
                llong( m ), llong( n ), llong( ld ) );
    }
    if (verbose >= 2) {
        printf( "a = " ); print_matrix( ld, n + extra, a_host, ld );
    }

    // run test
    testsweeper::flush_cache( params.cache() );

    //----------
    // a_host -> b_dev
    double time = sync_get_wtime( queue );
    if (method == Method::memcpy_2d) {
        blas::device_memcpy_2d( b_dev, ld, a_host, ld, m, n, queue );
    }
    else if (method == Method::copy_matrix) {
        blas::device_copy_matrix( m, n, a_host, ld, b_dev, ld, queue );
    }
    else if (method == Method::set_matrix) {
        blas::device_copy_matrix( m, n, a_host, ld, b_dev, ld, queue );
    }
    time = sync_get_wtime( queue ) - time;

    //----------
    // b_dev -> c_dev
    double time2 = sync_get_wtime( queue );
    if (method == Method::memcpy_2d) {
        blas::device_memcpy_2d( c_dev, ld, b_dev, ld, m, n, queue );
    }
    else {
        // For method = copy_matrix or set_matrix, use copy_matrix.
        blas::device_copy_matrix( m, n, b_dev, ld, c_dev, ld, queue );
    }
    time2 = sync_get_wtime( queue ) - time2;

    //----------
    // c_dev -> d_host
    double time3 = sync_get_wtime( queue );
    if (method == Method::memcpy_2d) {
        blas::device_memcpy_2d( d_host, ld, c_dev, ld, m, n, queue );
    }
    else if (method == Method::copy_matrix) {
        blas::device_copy_matrix( m, n, c_dev, ld, d_host, ld, queue );
    }
    else if (method == Method::set_matrix) {
        blas::device_copy_matrix( m, n, c_dev, ld, d_host, ld, queue );
    }
    time3 = sync_get_wtime( queue ) - time3;

    //----------
    // a_host -> b_host
    double time4 = sync_get_wtime( queue );
    if (method == Method::memcpy_2d) {
        blas::device_memcpy_2d( b_host, ld, a_host, ld, m, n, queue );
    }
    else {
        // For method = copy_matrix or set_matrix, use copy_matrix.
        blas::device_copy_matrix( m, n, a_host, ld, b_host, ld, queue );
    }
    time4 = sync_get_wtime( queue ) - time4;

    //----------
    // b_host -> c_host
    double ref_time = sync_get_wtime( queue );
    lapack_lacpy( "g", m, n, b_host, ld, c_host, ld );
    ref_time = sync_get_wtime( queue ) - ref_time;

    // read m*n, write m*n
    double gbyte = blas::Gbyte<T>::copy_2d( m, n );

    params.time()     = time;
    params.time2()    = time2;
    params.time3()    = time3;
    params.time4()    = time4;
    params.ref_time() = ref_time;

    params.gbytes()     = gbyte / time;
    params.gbytes2()    = gbyte / time2;
    params.gbytes3()    = gbyte / time3;
    params.gbytes4()    = gbyte / time4;
    params.ref_gbytes() = gbyte / ref_time;

    if (verbose >= 2) {
        printf( "b = " ); print_matrix( ld, n + extra, b_host, ld );
        printf( "c = " ); print_matrix( ld, n + extra, c_host, ld );
        printf( "d = " ); print_matrix( ld, n + extra, d_host, ld );
        printf( "Note last %lld rows and last %lld cols should NOT be copied!\n",
                llong( ld - m ), llong( extra ) );
    }

    // check error
    blas::axpy( size, -1.0, a_host, 1, b_host, 1 );
    blas::axpy( size, -1.0, a_host, 1, c_host, 1 );
    blas::axpy( size, -1.0, a_host, 1, d_host, 1 );
    real_t dummy;
    real_t error = lapack_lange( "m", m, n, b_host, ld, &dummy )
                 + lapack_lange( "m", m, n, c_host, ld, &dummy )
                 + lapack_lange( "m", m, n, d_host, ld, &dummy );
    // Entries outside sub-matrix should NOT be copied.
    // For first n cols, check i = m+1 : ld.
    // For extra   cols, check i = 0   : ld.
    for (int64_t j = 0; j < n + extra; ++j) {
        for (int64_t i = (j < n ? m : 0); i < ld; ++i) {
            if (a_host[ i + j*ld ] == b_host[ i + j*ld ])
                error += 1;
            if (a_host[ i + j*ld ] == c_host[ i + j*ld ])
                error += 1;
            if (a_host[ i + j*ld ] == d_host[ i + j*ld ])
                error += 1;
        }
    }
    params.error() = error;
    params.okay() = (error == 0);  // copy must be exact

    blas::host_free_pinned( a_host, queue );
    blas::host_free_pinned( b_host, queue );
    blas::host_free_pinned( c_host, queue );
    blas::host_free_pinned( d_host, queue );
    blas::device_free( b_dev, queue );
    blas::device_free( c_dev, queue );
}

// -----------------------------------------------------------------------------
void test_memcpy_2d( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_memcpy_2d_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_memcpy_2d_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_memcpy_2d_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_memcpy_2d_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
