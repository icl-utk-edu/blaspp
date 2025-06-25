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
template <typename TA, typename TX>
void test_trsv_device_work( Params& params, bool run )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    using namespace testsweeper;
    using blas::Uplo, blas::Op, blas::Layout, blas::Diag, blas::max;
    using scalar_t = blas::scalar_type< TA, TX >;
    using real_t   = blas::real_type< scalar_t >;
    using std::abs, std::swap;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    blas::Op trans  = params.trans();
    blas::Diag diag = params.diag();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t device  = params.device();
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

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    int64_t lda = max( roundup( n, align ), 1 );
    size_t size_A = size_t(lda)*n;
    size_t size_x = max( (n - 1) * abs( incx ) + 1, 0 );
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TX* xref = new TX[ size_x ];

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TX* dx;

    dA = blas::device_malloc<TA>( size_A, queue );
    dx = blas::device_malloc<TX>( size_x, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    // set unused data to nan
    if (uplo == Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = 0; i < j; ++i)  // upper
                A( i, j ) = nan("");
    }
    else {
        for (int64_t j = 0; j < n; ++j)
            for (int64_t i = j+1; i < n; ++i)  // lower
                A( i, j ) = nan("");
    }

    // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
    // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
    // First, brute force positive definiteness.
    for (int64_t i = 0; i < n; ++i) {
        A( i, i ) += n;
    }
    int64_t info = 0;
    lapack_potrf( to_c_string( uplo ), n, A, lda, &info );
    require( info == 0 );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lantr( "f", to_c_string( uplo ), to_c_string( diag ),
                                 n, n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, std::abs(incx) );

    // if row-major, transpose A
    if (layout == Layout::RowMajor) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < j; ++i) {
                swap( A( i, j ), A( j, i ) );
            }
        }
    }

    blas::device_copy_matrix( n, n, A, lda, dA, lda, queue );
    blas::device_copy_vector( n, x, abs( incx ), dx, abs( incx ), queue );
    queue.sync();

    // test error exits
    assert_throw( blas::trsv( Layout(0), uplo,    trans, diag,     n, dA, lda, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    Uplo(0), trans, diag,     n, dA, lda, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    uplo,    Op(0), diag,     n, dA, lda, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    uplo,    trans, Diag(0),  n, dA, lda, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    uplo,    trans, diag,    -1, dA, lda, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    uplo,    trans, diag,     n, dA, n-1, dx, incx, queue ), blas::Error );
    assert_throw( blas::trsv( layout,    uplo,    trans, diag,     n, dA, lda, dx,    0, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                llong( n ), llong( lda ),  llong( size_A ), Anorm,
                llong( n ), llong( incx ), llong( size_x ), Xnorm );
    }
    if (verbose >= 2) {
        printf( "A = "    ); print_matrix( n, n, A, lda );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::trsv( layout, uplo, trans, diag, n, dA, lda, dx, incx, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::trsv( n );
    double gbyte = blas::Gbyte< scalar_t >::trsv( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    blas::device_copy_vector( n, dx, abs( incx ), x, abs( incx ), queue );
    queue.sync();

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_trsv( cblas_layout_const(layout),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    cblas_diag_const(diag),
                    n, A, lda, xref, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // check error compared to reference
        // treat x as 1 x n matrix with ld = incx; k = n is reduction dimension
        // alpha = 1, beta = 0.
        real_t error;
        bool okay;
        check_gemm( 1, n, n, scalar_t(1), scalar_t(0), Anorm, Xnorm, real_t(0),
                    xref, std::abs(incx), x, std::abs(incx), verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] xref;

    blas::device_free( dA, queue );
    blas::device_free( dx, queue );

    #undef A
}

// -----------------------------------------------------------------------------
void test_trsv_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_trsv_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_trsv_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_trsv_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_trsv_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
