// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

// -----------------------------------------------------------------------------
template< typename Tx, typename Ty >
void test_dot_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas;
    typedef scalar_type<Tx, Ty> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    char mode       = params.pointer_mode();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

    scalar_t  result_host;
    scalar_t* result = &result_host;
    scalar_t  result_cblas;

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    Tx* x    = new Tx[ size_x ];
    Tx* xref = new Tx[ size_x ];
    Ty* y    = new Ty[ size_y ];
    Ty* yref = new Ty[ size_y ];

    // device specifics
    blas::Queue queue(device, 0);
    Tx* dx;
    Ty* dy;

    dx = blas::device_malloc<Tx>(size_x, queue);
    dy = blas::device_malloc<Ty>(size_y, queue);
    if (mode == 'd') {
        result = blas::device_malloc<scalar_t>(1, queue);
        #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
        #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
        #endif
    }

    // test error exits
    assert_throw( blas::dot( -1, x, incx, y, incy, result, queue ), blas::Error );
    assert_throw( blas::dot(  n, x,    0, y, incy, result, queue ), blas::Error );
    assert_throw( blas::dot(  n, x, incx, y,    0, result, queue ), blas::Error );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, x, incx, xref, incx );
    cblas_copy( n, y, incy, yref, incy );

    blas::device_setvector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    blas::device_setvector(n, y, std::abs(incy), dy, std::abs(incy), queue);
    queue.sync();

    if (verbose >= 1) {
        //printf( "\n"
        //        "n=%5lld, incx=%5lld, sizex=%10lld,",
        //        " incy=%5lld, sizey=%10lld\n",
        //        (lld) n, (lld) incx, (lld) size_x, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::dot( n, dx, incx, dy, incy, result, queue );
    queue.sync();
    time = get_wtime() - time;

    if (mode == 'd') {
        device_memcpy( &result_host, result, 1, queue );
    }

    double gflop = Gflop<scalar_t>::dot( n );
    double gbyte = Gbyte<scalar_t>::dot( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_getvector(n, dx, std::abs(incx), x, std::abs(incx), queue);
    blas::device_getvector(n, dy, std::abs(incy), y, std::abs(incy), queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        result_cblas = cblas_dot( n, xref, incx, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        //if (verbose >= 2) {
        //    printf( "result0 = %3.2lld\n", (lld) result_cblas);
        //}

        // relative forward error:
        real_t error = std::abs( (result_cblas - result_host ) )
                           / (sqrt(n+1) * std::abs( result_cblas ) );
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
    delete[] y;
    delete[] yref;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
    if (mode == 'd')
        blas::device_free( result, queue );
}

// -----------------------------------------------------------------------------
void test_dot_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_dot_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_dot_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_dot_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_dot_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
