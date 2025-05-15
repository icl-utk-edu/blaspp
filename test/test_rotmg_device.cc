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
void test_rotmg_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::abs;
    using std::real;
    using std::imag;
    using real_t   = blas::real_type< T >;

    // get & mark input values
    int64_t n = params.dim.n();
    int64_t device  = params.device();

    // mark non-standard output values
    params.ref_time();

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
    T* d1 = new T[ n ];
    T* d1_ref = new T[ n ];
    T* d2 = new T[ n ];
    T* d2_ref = new T[ n ];
    T* x1 = new T[ n ];
    T* x1_ref = new T[ n ];
    T* y1 = new T[ n ];
    T* y1_ref = new T[ n ];
    T* ps = new T[ 5*n ];
    T* ps_ref = new T[ 5*n ];

    int64_t idist = 3;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, n, d1 );
    lapack_larnv( idist, iseed, n, d2 );
    lapack_larnv( idist, iseed, n, x1 );
    lapack_larnv( idist, iseed, n, y1 );
    lapack_larnv( idist, iseed, 5*n, ps );

    // device specifics
    blas::Queue queue( device );
    T* d_d1;
    T* d_d2;
    T* d_x1;
    T* d_y1;
    T* d_ps;

    d_d1 = blas::device_malloc<T>( n, queue );
    d_d2 = blas::device_malloc<T>( n, queue );
    d_x1 = blas::device_malloc<T>( n, queue );
    d_y1 = blas::device_malloc<T>( n, queue );
    d_ps = blas::device_malloc<T>( 5*n, queue );

    device_memcpy( d_d1, d1, n, queue );
    device_memcpy( d_d2, d2, n, queue );
    device_memcpy( d_x1, x1, n, queue );
    device_memcpy( d_y1, y1, n, queue );
    device_memcpy( d_ps, ps, 5*n, queue );

    #if defined( BLAS_HAVE_CUBLAS )
        cublasSetPointerMode(queue.handle(), CUBLAS_POINTER_MODE_DEVICE);
    #elif defined( BLAS_HAVE_ROCBLAS )
        rocblas_set_pointer_mode( queue.handle(), rocblas_pointer_mode_device );
    #endif

    cblas_copy( n, d1, 1, d1_ref, 1 );
    cblas_copy( n, d2, 1, d2_ref, 1 );
    cblas_copy( n, x1, 1, x1_ref, 1 );
    cblas_copy( n, y1, 1, y1_ref, 1 );
    cblas_copy( 5*n, ps, 1, ps_ref, 1 );

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    for (int64_t i = 0; i < n; ++i) {
        blas::rotmg( d_d1 + i, d_d2 + i, d_x1 + i, d_y1 + i, d_ps + 5*i, queue );
    }
    queue.sync();
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

        device_memcpy( d1, d_d1, n, queue );
        device_memcpy( d2, d_d2, n, queue );
        device_memcpy( x1, d_x1, n, queue );
        device_memcpy( ps, d_ps, 5*n, queue );
        queue.sync();

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

    delete[] d1;
    delete[] d1_ref;
    delete[] d2;
    delete[] d2_ref;
    delete[] x1;
    delete[] x1_ref;
    delete[] y1;
    delete[] y1_ref;
    delete[] ps;
    delete[] ps_ref;

    blas::device_free( d_d1, queue );
    blas::device_free( d_d2, queue );
    blas::device_free( d_x1, queue );
    blas::device_free( d_y1, queue );
    blas::device_free( d_ps, queue );
}

// -----------------------------------------------------------------------------
void test_rotmg_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotmg_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotmg_device_work< double >( params, run );
            break;

        // modified Givens not available for complex

        default:
            throw std::exception();
            break;
    }
}
