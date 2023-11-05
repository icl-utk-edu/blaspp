// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename TX, typename TY>
void test_dot_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using scalar_t = blas::scalar_type< TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
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
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    TX* x = new TX[ size_x ];
    TY* y = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::dot( -1, x, incx, y, incy ), blas::Error );
    assert_throw( blas::dot(  n, x,    0, y, incy ), blas::Error );
    assert_throw( blas::dot(  n, x, incx, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n",
                llong( n ), llong( incx ), llong( size_x ), Xnorm,
                llong( n ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    scalar_t result = blas::dot( n, x, incx, y, incy );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::dot( n );
    double gbyte = blas::Gbyte< scalar_t >::dot( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 1) {
        printf( "dot = %.4e + %.4ei\n", real(result), imag(result) );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        scalar_t ref = cblas_dot( n, x, incx, y, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "ref = %.4e + %.4ei\n", real(ref), imag(ref) );
        }

        // check error compared to reference
        // treat result as 1 x 1 matrix; k = n is reduction dimension
        // alpha=1, beta=0, Cnorm=0
        real_t error;
        bool okay;
        check_gemm( 1, 1, n, scalar_t(1), scalar_t(0), Xnorm, Ynorm, real_t(0),
                    &ref, 1, &result, 1, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_dot( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_dot_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_dot_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_dot_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_dot_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
