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
template <typename TA, typename TB, typename TC>
void test_symm_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using std::real;
    using std::imag;
    using blas::Uplo;
    using blas::Side;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side     = params.side();
    blas::Uplo uplo     = params.uplo();
    scalar_t alpha      = params.alpha();
    scalar_t beta       = params.beta();
    int64_t m           = params.dim.m();
    int64_t n           = params.dim.n();
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
    int64_t An = (side == Side::Left ? m : n);
    int64_t Cm = m;
    int64_t Cn = n;
    if (layout == Layout::RowMajor)
        std::swap( Cm, Cn );
    int64_t lda = roundup( An, align );
    int64_t ldb = roundup( Cm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Cn;
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

    blas::device_copy_matrix(An, An, A, lda, dA, lda, queue);
    blas::device_copy_matrix(Cm, Cn, B, ldb, dB, ldb, queue);
    blas::device_copy_matrix(Cm, Cn, C, ldc, dC, ldc, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Cm, Cn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::symm( Layout(0), side,     uplo,     m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( layout,    Side(0),  uplo,     m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( layout,    side,     Uplo(0),  m,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( layout,    side,     uplo,    -1,  n, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( layout,    side,     uplo,     m, -1, alpha, dA, lda, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::symm( layout, Side::Left,  uplo,     m,  n, alpha, dA, m-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( layout, Side::Right, uplo,     m,  n, alpha, dA, n-1, dB, ldb, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::symm( Layout::ColMajor, side, uplo,  m,  n, alpha, dA, lda, dB, m-1, beta, dC, ldc, queue ), blas::Error );
    assert_throw( blas::symm( Layout::RowMajor, side, uplo,  m,  n, alpha, dA, lda, dB, n-1, beta, dC, ldc, queue ), blas::Error );

    assert_throw( blas::symm( Layout::ColMajor, side, uplo,  m,  n, alpha, dA, lda, dB, ldb, beta, dC, m-1, queue ), blas::Error );
    assert_throw( blas::symm( Layout::RowMajor, side, uplo,  m,  n, alpha, dA, lda, dB, ldb, beta, dC, n-1, queue ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "side %c, uplo %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B  m=%5lld,  n=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  m=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                side2char(side), uplo2char(uplo),
                llong( An ), llong( An ), llong( lda ), llong( size_A ), Anorm,
                llong( m ), llong( n ), llong( ldb ), llong( size_B ), Bnorm,
                llong( m ), llong( n ), llong( ldc ), llong( size_C ), Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( An, An, A, lda );
        printf( "B = "    ); print_matrix( Cm, Cn, B, ldb );
        printf( "C = "    ); print_matrix( Cm, Cn, C, ldc );
    }

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::symm( layout, side, uplo, m, n,
                alpha, dA, lda, dB, ldb, beta, dC, ldc, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = blas::Gflop< scalar_t >::symm( side, m, n );
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
        cblas_symm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    m, n, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
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

// -----------------------------------------------------------------------------
void test_symm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_symm_device_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_symm_device_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_symm_device_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_symm_device_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
