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
void test_rotg_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using std::real;
    using std::imag;
    using real_t   = blas::real_type< T >;

    // get & mark input values
    int64_t n = params.dim.n();
    int verbose = params.verbose();

    // mark non-standard output values
    params.ref_time();
    params.error2();
    params.error3();

    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    // setup
    std::vector<T> a( n ), aref( n ), a_in( n );
    std::vector<T> b( n ), bref( n ), b_in( n );
    std::vector<T> s( n ), sref( n );
    std::vector<real_t> c( n ), cref( n );

    int64_t idist = 3;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, n, &a[0] );
    lapack_larnv( idist, iseed, n, &b[0] );
    aref = a;
    bref = b;

    if (verbose >= 2) {
        printf( "a_in  = " );  print_vector( n, &a[0], 1 );
        printf( "b_in  = " );  print_vector( n, &b[0], 1 );
    }

    // Save some data to check later
    if (params.check() == 'y') {
        cblas_copy( n, &a[0], 1, &a_in[0], 1 );
        cblas_copy( n, &b[0], 1, &b_in[0], 1 );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    for (int64_t i = 0; i < n; ++i) {
        blas::rotg( &a[i], &b[i], &c[i], &s[i] );
    }
    time = get_wtime() - time;
    params.time() = time * 1000;  // msec

    if (verbose >= 2) {
        printf( "a_out = " );  print_vector( n, &a[0], 1 );
        printf( "b_out = " );  print_vector( n, &b[0], 1 );
        printf( "c     = " );  print_vector( n, &c[0], 1 );
        printf( "s     = " );  print_vector( n, &s[0], 1 );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (int64_t i = 0; i < n; ++i) {
            cblas_rotg( &aref[i], &bref[i], &cref[i], &sref[i] );
        }
        time = get_wtime() - time;
        params.ref_time() = time * 1000;  // msec

        if (verbose >= 2) {
            printf( "a_ref = " );  print_vector( n, &aref[0], 1 );
            printf( "b_ref = " );  print_vector( n, &bref[0], 1 );
            printf( "c_ref = " );  print_vector( n, &cref[0], 1 );
            printf( "s_ref = " );  print_vector( n, &sref[0], 1 );
        }

        // get max error of all outputs
        cblas_axpy( n, -1.0, &a[0], 1, &aref[0], 1 );
        cblas_axpy( n, -1.0, &b[0], 1, &bref[0], 1 );
        cblas_axpy( n, -1.0, &c[0], 1, &cref[0], 1 );
        cblas_axpy( n, -1.0, &s[0], 1, &sref[0], 1 );

        int64_t ia = cblas_iamax( n, &aref[0], 1 );
        int64_t ib = cblas_iamax( n, &bref[0], 1 );
        int64_t ic = cblas_iamax( n, &cref[0], 1 );
        int64_t is = cblas_iamax( n, &sref[0], 1 );

        real_t error = blas::max(
            abs( aref[ ia ] ),
            abs( bref[ ib ] ),
            abs( cref[ ic ] ),
            abs( sref[ is ] )
        );

        // error is normally 0, but allow for some rounding just in case.
        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < 10*u);

        // Applying the rotations $\begin{bmatrix} c & s \\ -s & c \end{bmatrix}$
        // to the vectors $[ a_i, b_i ]^T$
        // Expected output: $[ \sqrt{ a_i^2 + b_i^2 } & 0 ]$

        #define ABSSQ(t_) real(t_)*real(t_) + imag(t_)*imag(t_)

        std::vector<real_t> diffNorm2( n );
        for (int64_t i = 0; i < n; ++i) {
            diffNorm2[i] = ABSSQ( a_in[i] ) + ABSSQ( b_in[i] );
            blas::rot( 1, &a_in[i], 1, &b_in[i], 1, c[i], s[i] );
            diffNorm2[i] -= ABSSQ( a_in[i] );
        }

        #undef ABSSQ

        params.error2() = cblas_nrm2( n, &b_in[0], 1 );
        if (verbose >= 2) {
            printf( "null vector = " );  print_vector( n, &b_in[0], 1 );
        }

        params.error3() = cblas_nrm2( n, &diffNorm2[0], 1 );
        if (verbose >= 2) {
            printf( "null vector = " );  print_vector( n, &diffNorm2[0], 1 );
        }
    }
}

// -----------------------------------------------------------------------------
void test_rotg( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotg_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotg_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_rotg_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_rotg_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
