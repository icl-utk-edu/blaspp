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

//------------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_her2k_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo, blas::Op, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    real_t beta     = params.beta.get<real_t>();   // note: real
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // setup
    int64_t Am = (trans == Op::NoTrans ? n : k);
    int64_t An = (trans == Op::NoTrans ? k : n);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda = max( roundup( Am, align ), 1 );
    int64_t ldb = max( roundup( Am, align ), 1 );
    int64_t ldc = max( roundup(  n, align ), 1 );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*An;
    size_t size_C = size_t(ldc)*n;
    TA* A    = new TA[ size_A ];
    TB* B    = new TB[ size_B ];
    TC* C    = new TC[ size_C ];
    TC* Cref = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", n, n, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Am, An, B, ldb, work );
    real_t Cnorm = lapack_lansy( "f", to_c_string( uplo ), n, C, ldc, work );

    // test error exits
    assert_throw( blas::her2k( Layout(0), uplo,    trans,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( layout,    Uplo(0), trans,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( layout,    uplo,    Op(0),  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( layout,    uplo,    trans, -1,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( layout,    uplo,    trans,  n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::her2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::her2k( layout,    uplo,    trans,  n,  k, alpha, A, lda, B, ldb, beta, C, n-1 ), blas::Error );

    if (blas::is_complex_v<scalar_t>) {
        // complex her2k doesn't allow Trans, only ConjTrans
        assert_throw( blas::her2k( layout, uplo, Op::Trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    }

    if (verbose) {
        printf( "Anorm %.2e; Bnorm %.2e; Cnorm %.2e;\n",
                Anorm, Bnorm, Cnorm );
        print_matrix( "A", Am, An, A, lda, verbose );
        print_matrix( "B", Am, An, B, ldb, verbose );
        print_matrix( "C", n,  n, C, ldc, verbose );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::her2k( layout, uplo, trans, n, k,
                 alpha, A, lda, B, ldb, beta, C, ldc );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::her2k( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    print_matrix( "Cout", n, n, C, ldc, verbose );

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_her2k( cblas_layout_const(layout),
                     cblas_uplo_const(uplo),
                     cblas_trans_const(trans),
                     n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        print_matrix( "Cref", n, n, Cref, ldc, verbose );

        // check error compared to reference
        real_t error;
        bool okay;
        check_herk( uplo, n, 2*k, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
}

//------------------------------------------------------------------------------
void test_her2k( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_her2k_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_her2k_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_her2k_work< std::complex<float>, std::complex<float>,
                             std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_her2k_work< std::complex<double>, std::complex<double>,
                             std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
