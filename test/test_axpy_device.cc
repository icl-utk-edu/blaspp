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
template< typename T >
void test_axpy_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas;
    typedef real_type<T> real_t;
    typedef long long lld;

    // get & mark input values
    T alpha         = params.alpha();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

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
    T* x    = new T[ size_x ];
    T* xref = new T[ size_x ];
    T* y    = new T[ size_y ];
    T* yref = new T[ size_y ];

    // device specifics
    blas::Queue queue(device, 0);
    T* dx;
    T* dy;

    dx = blas::device_malloc<T>(size_x, queue);
    dy = blas::device_malloc<T>(size_y, queue);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );

    blas::device_setvector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    blas::device_setvector(n, y, std::abs(incy), dy, std::abs(incy), queue);
    queue.sync();

    // test error exits
    assert_throw( blas::axpy( -1, alpha, x, incx, y, incy, queue ), blas::Error );
    assert_throw( blas::axpy(  n, alpha, x,    0, y,    0, queue ), blas::Error );
    assert_throw( blas::axpy(  n, alpha, x,   -1, y,   -1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "n=%5lld, incx=%5lld, sizex=%10lld, incy=%5lld, sizey=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::axpy( n, alpha, dx, incx, dy, incy, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = Gflop < T >::axpy( n );
    double gbyte = Gbyte < T >::axpy( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_getvector(n, dx, std::abs(incx), x, std::abs(incx), queue);
    queue.sync();

    blas::device_getvector(n, dy, std::abs(incy), y, std::abs(incy), queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
        printf( "y2   = " ); print_vector( n, y, incy ); // TODO don't think needed
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_axpy( n, alpha, xref, incx, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // maximum component-wise forward error:
        // | fl(xi) - xi | / | xi |
        real_t error = 0;
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            error = std::max( error, std::abs( (xref[ix] - x[ix]) / xref[ix] ));
            ix += incx;
        }
        params.error() = error;
        // TODO add check on y?

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<T>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.okay() = (error < u);
    }

    delete[] x;
    delete[] xref;
    delete[] y;
    delete[] yref;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

// -----------------------------------------------------------------------------
void test_axpy_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_axpy_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_axpy_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_axpy_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_axpy_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
