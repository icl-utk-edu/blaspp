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
void test_scal_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::max;
    using std::abs;
    using std::real;
    using std::imag;
    using real_t = blas::real_type< T >;

    // get & mark input values
    T alpha         = params.alpha();
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
    T* x    = new T[ size_x ];
    T* xref = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    // test error exits
    assert_throw( blas::scal( -1, alpha, x, incx ), blas::Error );
    assert_throw( blas::scal(  n, alpha, x,    0 ), blas::Error );
    assert_throw( blas::scal(  n, alpha, x,   -1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::scal( n, alpha, x, incx );
    time = get_wtime() - time;

    double gflop = blas::Gflop< T >::scal( n );
    double gbyte = blas::Gbyte< T >::scal( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_scal( n, alpha, xref, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // maximum component-wise forward error:
        // | fl(xi) - xi | / | xi |
        real_t error = 0;
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            error = max( error, abs( (xref[ix] - x[ix]) / xref[ix] ));
            ix += incx;
        }
        params.error() = error;

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<T>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.okay() = (error < u);
    }

    delete[] x;
    delete[] xref;
}

// -----------------------------------------------------------------------------
void test_scal( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_scal_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_scal_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_scal_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_scal_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
