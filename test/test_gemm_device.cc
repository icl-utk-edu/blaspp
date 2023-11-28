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
void test_gemm_device_work( Params& params, bool run )
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
    blas::Op transA     = params.transA();
    blas::Op transB     = params.transB();
    scalar_t alpha      = params.alpha();
    scalar_t beta       = params.beta();
    int64_t m           = params.dim.m();
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

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TB* dB;
    TC* dC;

    dA = blas::device_malloc<TA>( size_A, queue );
    dB = blas::device_malloc<TB>( size_B, queue );
    dC = blas::device_malloc<TC>( size_C, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    blas::device_copy_matrix(Am, An, A, lda, dA, lda, queue);
    blas::device_copy_matrix(Bm, Bn, B, ldb, dB, ldb, queue);
    blas::device_copy_matrix(Cm, Cn, C, ldc, dC, ldc, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::gemm( Layout(0), transA, transB,  m,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    Op(0),  transB,  m,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, Op(0),   m,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB, -1,  n,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m, -1,  k, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m,  n, -1, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, dA, m-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, dA, k-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, dA, m-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, dA, m-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, dA, lda, B, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, dA, lda, B, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, dA, lda, B, n-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, dA, lda, B, n-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, dA, lda, B, k-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, dA, lda, B, k-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, m-1, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, dA, lda, dB, ldb, beta, dC, n-1, queue ), blas::Error );

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
                alpha, dA, lda, dB, ldb, beta, dC, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::gemm( m, n, k );
    params.time()   = time;
    params.gflops() = gflop / time;
    blas::device_copy_matrix(Cm, Cn, dC, ldc, C, ldc, queue);
    queue.sync();

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

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
    blas::device_free( dC, queue );
}
//
// -----------------------------------------------------------------------------
template <>
void test_gemm_device_work<blas::float16,blas::float16,blas::float16>( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Op;
    using blas::Layout;
    using scalar_hi = float;
    using scalar_lo = blas::float16;
    using real_t   = blas::real_type< scalar_hi >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op transA     = params.transA();
    blas::Op transB     = params.transB();
    scalar_lo alpha     = params.alpha();
    scalar_lo beta      = params.beta();
    int64_t m           = params.dim.m();
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
    blas::float16* A_lo   = new blas::float16[ size_A ];
    blas::float16* B_lo   = new blas::float16[ size_B ];
    blas::float16* C_lo   = new blas::float16[ size_C ];
    float* A_hi  = new float[ size_A ];
    float* B_hi  = new float[ size_B ];
    float* C_hi  = new float[ size_C ];
    float* Cref  = new float[ size_C ];

    // device specifics
    blas::Queue queue( device );
    blas::float16* dA_lo;
    blas::float16* dB_lo;
    blas::float16* dC_lo;
    float* dA_hi;
    float* dB_hi;
    float* dC_hi;

    dA_lo = blas::device_malloc<blas::float16>( size_A, queue );
    dB_lo = blas::device_malloc<blas::float16>( size_B, queue );
    dC_lo = blas::device_malloc<blas::float16>( size_C, queue );
    dA_hi = blas::device_malloc<float>( size_A, queue );
    dB_hi = blas::device_malloc<float>( size_B, queue );
    dC_hi = blas::device_malloc<float>( size_C, queue );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A_hi );
    lapack_larnv( idist, iseed, size_B, B_hi );
    lapack_larnv( idist, iseed, size_C, C_hi );
    lapack_lacpy( "g", Cm, Cn, C_hi, ldc, Cref, ldc );

    blas::device_copy_matrix( Am, An, A_hi, lda, dA_hi, lda, queue );
    blas::device_copy_matrix( Bm, Bn, B_hi, ldb, dB_hi, ldb, queue );
    blas::device_copy_matrix( Cm, Cn, C_hi, ldc, dC_hi, ldc, queue );

    // Convert float->float16
    blas::copy_matrix( Am, An, dA_hi, lda, dA_lo, lda, queue );
    blas::copy_matrix( Bm, Bn, dB_hi, ldb, dB_lo, ldb, queue );
    blas::copy_matrix( Cm, Cn, dC_hi, ldc, dC_lo, ldc, queue );
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A_hi, lda, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B_hi, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C_hi, ldc, work );

    // test error exits
    assert_throw( blas::gemm( Layout(0), transA, transB,  m,  n,  k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    Op(0),  transB,  m,  n,  k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, Op(0),   m,  n,  k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB, -1,  n,  k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m, -1,  k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( layout,    transA, transB,  m,  n, -1, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, dA_hi, m-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, dA_hi, k-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, dA_hi, k-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans,   Op::NoTrans, m, n, k, alpha, dA_hi, k-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::Trans,     Op::NoTrans, m, n, k, alpha, dA_hi, m-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::ConjTrans, Op::NoTrans, m, n, k, alpha, dA_hi, m-1, dB_hi, ldb, beta, dC_hi, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, dA_hi, lda, dB_hi, k-1, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, dA_hi, lda, dB_hi, n-1, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::ColMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, dA_hi, lda, dB_hi, n-1, beta, dC_hi, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::NoTrans,   m, n, k, alpha, dA_hi, lda, dB_hi, n-1, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::Trans,     m, n, k, alpha, dA_hi, lda, dB_hi, k-1, beta, dC_hi, ldc, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, Op::NoTrans, Op::ConjTrans, m, n, k, alpha, dA_hi, lda, dB_hi, k-1, beta, dC_hi, ldc, queue ), blas::Error );

    assert_throw( blas::gemm( Layout::ColMajor, transA, transB, m, n, k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, m-1, queue ), blas::Error );
    assert_throw( blas::gemm( Layout::RowMajor, transA, transB, m, n, k, alpha, dA_hi, lda, dB_hi, ldb, beta, dC_hi, n-1, queue ), blas::Error );

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
                blas::real(alpha), blas::imag(alpha),
                blas::real(beta),  blas::imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A_hi, lda );
        printf( "B = "    ); print_matrix( Bm, Bn, B_hi, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C_hi, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::gemm( layout, transA, transB, m, n, k,
                alpha, dA_lo, lda, dB_lo, ldb, beta, dC_lo, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_lo >::gemm( m, n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    // Convert float16->float
    blas::copy_matrix( Cm, Cn, dC_lo, ldc, dC_hi, ldc, queue );
    blas::device_copy_matrix(Cm, Cn, dC_hi, ldc, C_hi, ldc, queue);
    queue.sync();

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
                    m, n, k, alpha, A_hi, lda, B_hi, ldb, beta, Cref, ldc ); // keep it like this as it defines the reference
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

    delete[] A_hi;
    delete[] B_hi;
    delete[] C_hi;
    delete[] A_lo;
    delete[] B_lo;
    delete[] C_lo;
    delete[] Cref;

    blas::device_free( dA_hi, queue );
    blas::device_free( dB_hi, queue );
    blas::device_free( dC_hi, queue );
    blas::device_free( dA_lo, queue );
    blas::device_free( dB_lo, queue );
    blas::device_free( dC_lo, queue );
}

// -----------------------------------------------------------------------------
void test_gemm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Half:
            test_gemm_device_work< blas::float16, blas::float16, blas::float16 >( params, run );
            break;

        case testsweeper::DataType::Single:
            test_gemm_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_gemm_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_gemm_device_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_gemm_device_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
