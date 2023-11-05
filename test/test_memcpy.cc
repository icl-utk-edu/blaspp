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
void test_memcpy_work( Params& params, bool run )
{
    using namespace testsweeper;
    using real_t = blas::real_type<T>;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t inc_ac  = params.incx();
    int64_t inc_bd  = params.incy();
    int64_t device  = params.device();
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
        memcpy,
        copy_vector,
        set_vector,   // tests both set_vector and get_vector
    };

    Method method;
    if (params.routine == "memcpy") {
        method = Method::memcpy;
        blas_error_if_msg( inc_ac != 1, "memcpy doesn't have inc" );
        blas_error_if_msg( inc_bd != 1, "memcpy doesn't have inc" );
    }
    else if (params.routine == "copy_vector") {
        method = Method::copy_vector;
    }
    else if (params.routine == "set_vector") {
        method = Method::set_vector;
    }
    else {
        throw blas::Error( "unknown method" );
    }

    // setup
    if (verbose >= 1)
        printf( "setup\n" );

    blas::Queue queue( device );

    // Allocate extra to verify copy doesn't overrun buffer.
    // When routine is copy/set/get_vector,
    // inc_ac is used for a, c vectors,
    // inc_bd is used for b, d vectors.
    int64_t inc = std::max( inc_ac, inc_bd );
    int64_t extra = 2;
    int64_t size = inc*(n + extra);
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
    if (verbose >= 1)
        printf( "error exits\n" );

    assert_throw( blas::device_memcpy( c_dev, b_dev, -1, queue ), blas::Error );

    assert_throw( blas::device_copy_vector( -1, b_dev, 1, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 0, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 1, c_dev, 0, queue ), blas::Error );

    assert_throw( blas::device_copy_vector( -1, b_dev, 1, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 0, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 1, c_dev, 0, queue ), blas::Error );

    assert_throw( blas::device_copy_vector( -1, b_dev, 1, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 0, c_dev, 1, queue ), blas::Error );
    assert_throw( blas::device_copy_vector(  n, b_dev, 1, c_dev, 0, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "n=%5lld, inc_ac=%2lld, inc_bd=%2lld\n",
                llong( n ), llong( inc_ac ), llong( inc_bd ) );
    }
    if (verbose >= 2) {
        printf( "a = " ); print_matrix( inc_ac, n + extra, a_host, inc_ac );
    }

    // run test
    testsweeper::flush_cache( params.cache() );

    //----------
    // a_host -> b_dev
    if (verbose >= 1)
        printf( "a_host -> b_dev,  method %d\n", int( method ) );

    double time = sync_get_wtime( queue );
    if (method == Method::memcpy) {
        blas::device_memcpy( b_dev, a_host, n, queue );
    }
    else if (method == Method::copy_vector) {
        blas::device_copy_vector( n, a_host, inc_ac, b_dev, inc_bd, queue );
    }
    else if (method == Method::set_vector) {
        blas::device_copy_vector( n, a_host, inc_ac, b_dev, inc_bd, queue );
    }
    time = sync_get_wtime( queue ) - time;

    //----------
    // b_dev -> c_dev
    if (verbose >= 1)
        printf( "b_dev  -> c_dev,  method %d\n", int( method ) );

    double time2 = sync_get_wtime( queue );
    if (method == Method::memcpy) {
        blas::device_memcpy( c_dev, b_dev, n, queue );
    }
    else {
        // For method = copy_vector or set_vector, use copy_vector.
        blas::device_copy_vector( n, b_dev, inc_bd, c_dev, inc_ac, queue );
    }
    time2 = sync_get_wtime( queue ) - time2;

    //----------
    // c_dev -> d_host
    if (verbose >= 1)
        printf( "c_dev  -> d_host, method %d\n", int( method ) );

    double time3 = sync_get_wtime( queue );
    if (method == Method::memcpy) {
        blas::device_memcpy( d_host, c_dev, n, queue );
    }
    else if (method == Method::copy_vector) {
        blas::device_copy_vector( n, c_dev, inc_ac, d_host, inc_bd, queue );
    }
    else if (method == Method::set_vector) {
        blas::device_copy_vector( n, c_dev, inc_ac, d_host, inc_bd, queue );
    }
    time3 = sync_get_wtime( queue ) - time3;

    //----------
    // a_host -> b_host
    if (verbose >= 1)
        printf( "a_host -> b_host, method %d\n", int( method ) );

    double time4 = sync_get_wtime( queue );
    if (method == Method::memcpy) {
        blas::device_memcpy( b_host, a_host, n, queue );
    }
    else {
        // For method = copy_vector or set_vector, use copy_vector.
        blas::device_copy_vector( n, a_host, inc_ac, b_host, inc_bd, queue );
    }
    time4 = sync_get_wtime( queue ) - time4;

    //----------
    // b_host -> c_host
    if (verbose >= 1)
        printf( "b_host -> c_host, method %d\n", int( method ) );

    double ref_time = sync_get_wtime( queue );
    blas::copy( n, b_host, inc_bd, c_host, inc_ac );
    ref_time = sync_get_wtime( queue ) - ref_time;

    // read n, write n
    double gbyte = blas::Gbyte<T>::copy( n );

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
        printf( "b = " ); print_matrix( inc_bd, n + extra, b_host, inc_bd );
        printf( "c = " ); print_matrix( inc_ac, n + extra, c_host, inc_ac );
        printf( "d = " ); print_matrix( inc_bd, n + extra, d_host, inc_bd );
        printf( "Note rows after first and last %lld cols should NOT be copied!\n",
                llong( extra ) );
    }

    // check error
    blas::axpy( n, -1.0, a_host, inc_ac, b_host, inc_bd );
    blas::axpy( n, -1.0, a_host, inc_ac, c_host, inc_ac );
    blas::axpy( n, -1.0, a_host, inc_ac, d_host, inc_bd );
    // Interpret as one row of inc-by-n matrix.
    real_t dummy;
    real_t error = lapack_lange( "m", 1, n, b_host, inc_bd, &dummy )
                 + lapack_lange( "m", 1, n, c_host, inc_ac, &dummy )
                 + lapack_lange( "m", 1, n, d_host, inc_bd, &dummy );
    // Entries outside increments should NOT be copied.
    // For first n cols, check i = 1 : inc.
    // For extra   cols, check i = 0 : inc.
    int64_t min_inc = std::min( inc_ac, inc_bd );
    for (int64_t j = 0; j < n + extra; ++j) {
        for (int64_t i = (j < n ? 1 : 0); i < min_inc; ++i) {
            if (a_host[ i + j*inc_ac ] == b_host[ i + j*inc_bd ])
                error += 1;
            if (a_host[ i + j*inc_ac ] == c_host[ i + j*inc_ac ])
                error += 1;
            if (a_host[ i + j*inc_ac ] == d_host[ i + j*inc_bd ])
                error += 1;
        }
    }
    params.error() = error;
    params.okay() = (error == 0);  // copy must be exact

    if (verbose >= 1)
        printf( "cleanup\n" );

    blas::host_free_pinned( a_host, queue );
    blas::host_free_pinned( b_host, queue );
    blas::host_free_pinned( c_host, queue );
    blas::host_free_pinned( d_host, queue );
    blas::device_free( b_dev, queue );
    blas::device_free( c_dev, queue );
}

// -----------------------------------------------------------------------------
void test_memcpy( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_memcpy_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_memcpy_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_memcpy_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_memcpy_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
