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

#include "utils.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_gemm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA = params.transA();
    blas::Op transB = params.transB();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
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
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
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

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C Cm=%5lld, Cn=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm,
                llong( Cm ), llong( Cn ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
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

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

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

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

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

// -----------------------------------------------------------------------------
template <>
void test_gemm_work<blas::float16, blas::float16, blas::float16>(
    Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::real;
    using blas::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_hi = float;
    using scalar_lo = blas::float16;
    using real_t   = blas::real_type< scalar_hi >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA  = params.transA();
    blas::Op transB  = params.transB();
    scalar_lo alpha  = (scalar_lo)params.alpha();
    scalar_lo beta   = (scalar_lo)params.beta();
    int64_t m        = params.dim.m();
    int64_t n        = params.dim.n();
    int64_t k        = params.dim.k();
    int64_t align    = params.align();
    int64_t verbose  = params.verbose();

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
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    blas::float16* A_lo    = new blas::float16[ size_A ];
    blas::float16* B_lo    = new blas::float16[ size_B ];
    blas::float16* C_lo    = new blas::float16[ size_C ];
    float* A_hi = new float[ size_A ];
    float* B_hi = new float[ size_B ];
    float* C_hi = new float[ size_C ];
    float* Cref = new float[ size_C ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A_hi );
    lapack_larnv( idist, iseed, size_B, B_hi );
    lapack_larnv( idist, iseed, size_C, C_hi );
    lapack_lacpy( "g", Cm, Cn, C_hi, ldc, Cref, ldc );

    // Convert float->float16
    blas::copy_matrix( Am, An, A_hi, lda, A_lo, lda );
    blas::copy_matrix( Bm, Bn, B_hi, ldb, B_lo, ldb );
    blas::copy_matrix( Cm, Cn, C_hi, ldc, C_lo, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A_hi, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B_hi, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C_hi, ldc, work );

    // test error exits
    assert_throw( blas::gemm( Layout(0), transA, transB,  m,  n,  k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    Op(0),  transB,  m,  n,  k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, Op(0),   m,  n,  k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB, -1,  n,  k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m, -1,  k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m,  n, -1, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A_hi, m-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A_hi, k-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A_hi, k-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, A_hi, k-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, A_hi, m-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, A_hi, m-1, B_hi, ldb, beta, C_hi, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A_hi, lda, B_hi, k-1, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A_hi, lda, B_hi, n-1, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A_hi, lda, B_hi, n-1, beta, C_hi, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, A_hi, lda, B_hi, n-1, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, A_hi, lda, B_hi, k-1, beta, C_hi, ldc ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, A_hi, lda, B_hi, k-1, beta, C_hi, ldc ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, m-1 ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, A_hi, lda, B_hi, ldb, beta, C_hi, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bm=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C Cm=%5lld, Cn=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                llong( Am ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( Bm ), llong( Bn ), llong( ldb ), llong( size_B ), Bnorm,
                llong( Cm ), llong( Cn ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A_hi, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B_hi, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C_hi, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::gemm( layout, transA, transB, m, n, k,
                alpha, A_lo, lda, B_lo, ldb, beta, C_lo, ldc );
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_lo >::gemm( m, n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    // Convert float16->float
    blas::copy_matrix( Cm, Cn, C_lo, ldc, C_hi, ldc );

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C_hi, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        cblas_gemm( cblas_layout_const(layout),
                    cblas_trans_const(transA),
                    cblas_trans_const(transB),
                    m, n, k, alpha, A_hi, lda, B_hi, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm<float, blas::float16>(
            Cm, Cn, k, alpha, beta, Anorm, Bnorm, Cnorm,
            Cref, ldc, C_hi, ldc, verbose, &error, &okay );
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A_lo;
    delete[] B_lo;
    delete[] C_lo;
    delete[] A_hi;
    delete[] B_hi;
    delete[] C_hi;
    delete[] Cref;
}

// -----------------------------------------------------------------------------
void test_gemm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Half:
            test_gemm_work< blas::float16, blas::float16, blas::float16 >(
                params, run );
            break;

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
