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
template <typename Tx>
void test_nrm2_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using scalar_t = blas::scalar_type<Tx>;
    using real_t   = blas::real_type<scalar_t>;
    using std::abs;

    // get & mark input values
    char mode       = params.pointer_mode();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

    real_t  result_host;
    real_t* result = &result_host;
    real_t  result_cblas;

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
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    Tx* x    = new Tx[ size_x ];
    Tx* xref = new Tx[ size_x ];

    // device specifics
    blas::Queue queue( device );
    Tx* dx;

    dx = blas::device_malloc<Tx>(size_x, queue);
    if (mode == 'd') {
        result = blas::device_malloc<real_t>(1, queue);
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

    // test error exits
    assert_throw( blas::nrm2( -1, x, incx, result, queue ), blas::Error );
    assert_throw( blas::nrm2(  n, x,    0, result, queue ), blas::Error );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    if (verbose >= 1) {
        printf( "\n"
                "n=%5lld, incx=%5lld, sizex=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::nrm2( n, dx, incx, result, queue );
    queue.sync();
    time = get_wtime() - time;

    if (mode == 'd') {
        device_memcpy( &result_host, result, 1, queue );
    }

    double gflop = blas::Gflop< Tx >::nrm2( n );
    double gbyte = blas::Gbyte< Tx >::nrm2( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector(n, dx, std::abs(incx), x, std::abs(incx), queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        result_cblas = cblas_nrm2( n, xref, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "result0 = %.2e\n", result_cblas );
        }

        // relative forward error:
        real_t error = abs( (result_cblas - result_host)
                           / (sqrt(n+1) * result_cblas) );
        params.error() = error;


        if (verbose >= 2) {
            printf( "err  = " ); print_vector( n, x, incx, "%9.2e" );
        }

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<scalar_t>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < u);
    }

    delete[] x;
    delete[] xref;

    blas::device_free( dx, queue );
    if (mode == 'd')
        blas::device_free( result, queue );
}

// -----------------------------------------------------------------------------
void test_nrm2_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_nrm2_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_nrm2_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_nrm2_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_nrm2_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
