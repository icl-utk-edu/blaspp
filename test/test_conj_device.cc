// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "blas/device.hh"

//------------------------------------------------------------------------------
template <typename scalar_t>
void cpu_conj(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t*       y, int64_t incy )
{
    using blas::conj;

    int64_t ix = (incx > 0 ? 0 : (1 - n) * incx);
    int64_t iy = (incy > 0 ? 0 : (1 - n) * incy);

    for (int i = 0; i < n; ++i) {
        y[i * incy + iy] = conj( x[i * incx + ix] );
    }
}

//------------------------------------------------------------------------------
template <typename scalar_t>
void test_conj_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs, std::real, std::imag;
    using blas::max;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t device  = params.device();
    int64_t verbose = params.verbose();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    size_t size_y = max( (n - 1) * abs( incy ) + 1, 0 );
    scalar_t* x    = new scalar_t[ size_x ];
    scalar_t* y    = new scalar_t[ size_y ];
    scalar_t* yref = new scalar_t[ size_y ];

    // device specifics
    blas::Queue queue( device );
    scalar_t* dx;
    scalar_t* dy;

    dx = blas::device_malloc<scalar_t>( size_x, queue );
    dy = blas::device_malloc<scalar_t>( size_y, queue );
    queue.sync();

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    blas::device_copy_vector( n, y, abs( incy ), dy, abs( incy ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::conj( -1, x, incx, y, incy, queue ), blas::Error );
    assert_throw( blas::conj(  n, x,    0, y, incy, queue ), blas::Error );
    assert_throw( blas::conj(  n, x, incx, y,    0, queue ), blas::Error );

    print_vector( "x", n, x, incx, verbose );

    // run test
    testsweeper::flush_cache( params.cache() );
    blas::conj( n, dx, incx, dy, incy, queue );
    queue.sync();

    blas::device_copy_vector( n, dx, abs( incx ), x, abs( incx ), queue );
    blas::device_copy_vector( n, dy, abs( incy ), y, abs( incy ), queue );
    queue.sync();

    print_vector( "y", n, y, incy, verbose );

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        cpu_conj( n, x, incx, yref, incy );

        print_vector( "yref", n, yref, incy, verbose );

        // error = ||yref - y||
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, yref, abs( incy ) );
        params.error() = error;

        // result is expected to be identical since only sign changes
        params.okay() = (error == 0);
    }

    delete[] x;
    delete[] y;
    delete[] yref;

    blas::device_free( dx, queue );
    blas::device_free( dy, queue );
}

//------------------------------------------------------------------------------
void test_conj_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_conj_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_conj_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_conj_device_work< std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_conj_device_work< std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
