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
void test_gemm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op, blas::Layout, blas::max;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();
    scalar_t alpha  = params.alpha.get<scalar_t>();
    scalar_t beta   = params.beta.get<scalar_t>();
    int64_t m       = params.dim.m();
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
    int64_t Am = (transA == Op::NoTrans ? m : k);
    int64_t An = (transA == Op::NoTrans ? k : m);
    int64_t Bm = (transB == Op::NoTrans ? k : n);
    int64_t Bn = (transB == Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }
    int64_t lda = max( roundup( Am, align ), 1 );
    int64_t ldb = max( roundup( Bm, align ), 1 );
    int64_t ldc = max( roundup( Cm, align ), 1 );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    TA* A    = new TA[ size_A ];
    TB* B    = new TB[ size_B ];
    TC* C    = new TC[ size_C ];
    TC* Cref = new TC[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::gemm( Layout(0), transA, transB,  m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    Op(0),  transB,  m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, Op(0),   m,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB, -1,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m, -1,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m,  n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, m-1 ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, n-1 ), blas::Error );

    if (verbose) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "Anorm = %.2e; Bnorm = %.2e; Cnorm = %.2e;\n",
                Anorm, Bnorm, Cnorm );
        print_matrix( "A", Am, An, A, lda, verbose );
        print_matrix( "B", Bm, Bn, B, ldb, verbose );
        print_matrix( "C", Cm, Cn, C, ldc, verbose );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::gemm( m, n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    print_matrix( "Cout", Cm, Cn, C, ldc, verbose );

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_gemm( cblas_layout_const(layout),
                    cblas_trans_const(transA),
                    cblas_trans_const(transB),
                    m, n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        print_matrix( "Cref", Cm, Cn, Cref, ldc, verbose );

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, k, alpha, beta, Anorm, Bnorm, Cnorm,
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
void test_gemm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_gemm_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemm_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
