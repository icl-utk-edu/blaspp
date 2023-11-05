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
template <typename TA, typename TX, typename TY>
void test_gemv_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TX, TY >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t align   = params.align();
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
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    int64_t Xm = (trans == Op::NoTrans ? n : m);
    int64_t Ym = (trans == Op::NoTrans ? m : n);
    size_t size_A = size_t(lda)*An;
    size_t size_x = (Xm - 1) * std::abs(incx) + 1;
    size_t size_y = (Ym - 1) * std::abs(incy) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( Ym, y, incy, yref, incy );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( Xm, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( Ym, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::gemv( Layout(0), trans,  m,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( layout,    Op(0),  m,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( layout,    trans, -1,  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( layout,    trans,  m, -1, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );

    assert_throw( blas::gemv( Layout::ColMajor, trans,  m,  n, alpha, A, m-1, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( Layout::RowMajor, trans,  m,  n, alpha, A, n-1, x, incx, beta, y, incy ), blas::Error );

    assert_throw( blas::gemv( layout,    trans,  m,  n, alpha, A, lda, x, 0,    beta, y, incy ), blas::Error );
    assert_throw( blas::gemv( layout,    trans,  m,  n, alpha, A, lda, x, incx, beta, y, 0    ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x Xm=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n"
                "y Ym=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Xm ), llong( incx ), llong( size_x ), Xnorm,
                llong( Ym ), llong( incy ), llong( size_y ), Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( m, n, A, lda );
        printf( "x    = " ); print_vector( Xm, x, incx );
        printf( "y    = " ); print_vector( Ym, y, incy );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::gemv( layout, trans, m, n, alpha, A, lda, x, incx, beta, y, incy );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::gemv( m, n );
    double gbyte = blas::Gbyte< scalar_t >::gemv( m, n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( Ym, y, incy );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_gemv( cblas_layout_const(layout), cblas_trans_const(trans), m, n,
                    alpha, A, lda, x, incx, beta, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time; // * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( Ym, yref, incy );
        }

        // check error compared to reference
        // treat y as 1 x Ym matrix with ld = incy; k = Xm is reduction dimension
        real_t error;
        bool okay;
        check_gemm( 1, Ym, Xm, alpha, beta, Anorm, Xnorm, Ynorm,
                    yref, std::abs(incy), y, std::abs(incy), verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_gemv( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemv_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemv_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemv_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemv_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
