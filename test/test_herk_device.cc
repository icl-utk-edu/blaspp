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
template <typename TA, typename TC>
void test_herk_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Uplo;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans      = params.trans();
    blas::Uplo uplo     = params.uplo();
    real_t alpha        = params.alpha();  // note: real
    real_t beta         = params.beta();   // note: real
    int64_t n           = params.dim.n();
    int64_t k           = params.dim.k();
    int64_t device      = params.device();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = roundup( Am, align );
    int64_t ldc = roundup(  n, align );
    size_t size_A = size_t(lda)*An;
    size_t size_C = size_t(ldc)*n;
    TA* A    = new TA[ size_A ];
    TC* C    = new TC[ size_C ];
    TC* Cref = new TC[ size_C ];

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TC* dC;

    dA = blas::device_malloc<TA>( size_A, queue );
    dC = blas::device_malloc<TC>( size_C, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    blas::device_copy_matrix(Am, An, A, lda, dA, lda, queue);
    blas::device_copy_matrix(n,  n,  C, ldc, dC, ldc, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );

    // test error exits
    assert_throw( blas::herk( Layout(0), uplo,    trans,  n,  k, alpha, dA, lda, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( layout,    Uplo(0), trans,  n,  k, alpha, dA, lda, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( layout,    uplo,    Op(0),  n,  k, alpha, dA, lda, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( layout,    uplo,    trans, -1,  k, alpha, dA, lda, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( layout,    uplo,    trans,  n, -1, alpha, dA, lda, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::herk( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, dA, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, dA, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, dA, k-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::herk( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, dA, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, dA, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::herk( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, dA, n-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::herk( layout,    uplo,    trans,  n,  k, alpha, dA, lda, beta, dC, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( n ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e; beta = %.4e;  %% real\n", alpha, beta );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::herk( layout, uplo, trans, n, k,
                alpha, dA, lda, beta, dC, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::herk( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;
    blas::device_copy_matrix(n, n, dC, ldc, C, ldc, queue);
    queue.sync();

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( n, n, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_herk( cblas_layout_const(layout),
                    cblas_uplo_const(uplo),
                    cblas_trans_const(trans),
                    n, k, alpha, A, lda, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( n, n, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_herk( uplo, n, k, alpha, beta, Anorm, Anorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] C;
    delete[] Cref;

    blas::device_free( dA, queue );
    blas::device_free( dC, queue );
}

// -----------------------------------------------------------------------------
void test_herk_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_herk_device_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_herk_device_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_herk_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_herk_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
