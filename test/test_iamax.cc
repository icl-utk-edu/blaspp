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
void test_iamax_work( Params& params, bool run )
{
    using namespace testsweeper;
    using real_t   = blas::real_type< T >;
    using std::abs;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t verbose = params.verbose();

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

    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    T* x = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    // test error exits
    assert_throw( blas::iamax( -1, x, incx ), blas::Error );
    assert_throw( blas::iamax(  n, x,    0 ), blas::Error );
    assert_throw( blas::iamax(  n, x,   -1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    int64_t result = blas::iamax( n, x, incx );
    time = get_wtime() - time;

    double gflop = blas::Gflop< T >::iamax( n );
    double gbyte = blas::Gbyte< T >::iamax( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 1) {
        printf( "result = %5lld\n", llong( result ) );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        int64_t ref = cblas_iamax( n, x, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "ref    = %5lld\n", llong( ref ) );
        }

        // error = |ref - result|
        real_t error = abs( ref - result );
        params.error() = error;

        // iamax must be exact!
        params.okay() = (error == 0);
    }

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_iamax( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_iamax_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_iamax_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_iamax_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_iamax_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
