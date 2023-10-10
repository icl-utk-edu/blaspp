// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template <typename T>
void test_rotmg_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using std::real;
    using std::imag;
    using real_t   = blas::real_type< T >;

    // get & mark input values
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    // setup
    std::vector<T> d1( n ), d1_ref( n );
    std::vector<T> d2( n ), d2_ref( n );
    std::vector<T> x1( n ), x1_ref( n );
    std::vector<T> y1( n ), y1_ref( n );
    std::vector<T> ps( 5*n ), ps_ref( 5*n );

    int64_t idist = 3;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, n, &d1[0] );
    lapack_larnv( idist, iseed, n, &d2[0] );
    lapack_larnv( idist, iseed, n, &x1[0] );
    lapack_larnv( idist, iseed, n, &y1[0] );
    lapack_larnv( idist, iseed, 5*n, &ps[0] );

    d1_ref = d1;
    d2_ref = d2;
    x1_ref = x1;
    y1_ref = y1;
    ps_ref = ps;

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    for (int64_t i = 0; i < n; ++i) {
        blas::rotmg( &d1[i], &d2[i], &x1[i], y1[i], &ps[5*i] );
    }
    time = get_wtime() - time;
    params.time() = time * 1000;  // msec

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (int64_t i = 0; i < n; ++i) {
            cblas_rotmg( &d1_ref[i], &d2_ref[i], &x1_ref[i], y1_ref[i], &ps_ref[5*i] );
        }
        time = get_wtime() - time;
        params.ref_time() = time * 1000;  // msec

        // get max error of all outputs
        cblas_axpy(   n, -1.0, &d1[0], 1, &d1_ref[0], 1 );
        cblas_axpy(   n, -1.0, &d2[0], 1, &d2_ref[0], 1 );
        cblas_axpy(   n, -1.0, &x1[0], 1, &x1_ref[0], 1 );
        cblas_axpy( 5*n, -1.0, &ps[0], 1, &ps_ref[0], 1 );

        int64_t id1 = cblas_iamax(   n, &d1_ref[0], 1 );
        int64_t id2 = cblas_iamax(   n, &d2_ref[0], 1 );
        int64_t ix1 = cblas_iamax(   n, &x1_ref[0], 1 );
        int64_t ips = cblas_iamax( 5*n, &ps_ref[0], 1 );

        real_t error = blas::max(
            abs( d1_ref[ id1 ] ),
            abs( d2_ref[ id2 ] ),
            abs( x1_ref[ ix1 ] ),
            abs( ps_ref[ ips ] )
        );

        // error is normally 0, but allow for some rounding just in case.
        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < 10*u);
    }
}

// -----------------------------------------------------------------------------
void test_rotmg( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotmg_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotmg_work< double >( params, run );
            break;

        // modified Givens not available for complex

        default:
            throw std::exception();
            break;
    }
}
