// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template <typename T>
void test_iamax_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas;
    using scalar_t = blas::scalar_type< T >;
    using real_t   = blas::real_type< scalar_t >;
    using std::abs;
    using blas::max;

    // get & mark input values
    char mode = params.pointer_mode();
    int64_t n = params.dim.n();
    int64_t incx = params.incx();
    int64_t device = params.device();
    int64_t verbose = params.verbose();

    int64_t result_host;
    int64_t* result_ptr = &result_host;

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    T* x = new T[ size_x ];


    int64_t idist = 1;
    int iseed[ 4 ] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    // device specifics
    blas::Queue queue( device );
    T* dx;

    dx = blas::device_malloc<T>( size_x, queue );
    blas::device_copy_vector( n, x, std::abs(incx), dx, std::abs(incx), queue );
    queue.sync();

    if (mode == 'd') {
        result_ptr = blas::device_malloc< int64_t >( 1, queue );
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode( queue.handle(), CUBLAS_POINTER_MODE_DEVICE );
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

    // test error exits
    assert_throw( blas::iamax( -1, x, incx, result_ptr, queue ), blas::Error );
    assert_throw( blas::iamax(  n, x,    0, result_ptr, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::iamax( n, dx, incx, result_ptr, queue );
    queue.sync();
    time = get_wtime() - time;

    if (mode == 'd') {
        device_memcpy( &result_host, result_ptr, 1, queue );
    }

    double gflop = blas::Gflop< T >::iamax( n );
    double gbyte = blas::Gbyte< T >::iamax( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector( n, dx, std::abs(incx), x, std::abs(incx), queue );
    queue.sync();

    if (verbose >= 1) {
        printf( "result = %5lld\n", llong( result_host ) );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        int64_t ref = cblas_iamax( n, x, incx );
        if (n == 0)
            ref -= 1;
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "ref    = %5lld\n", llong( ref ) );
        }

        // check error compared to reference
        real_t error = abs( ref - result_host );
        params.error() = error;

        // iamax must be exact!
        params.okay() = (error == 0);
    }

    delete[] x;

    blas::device_free( dx, queue );
    if (mode == 'd')
        blas::device_free( result_ptr, queue );
}

// -----------------------------------------------------------------------------
void test_iamax_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_iamax_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_iamax_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_iamax_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_iamax_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
