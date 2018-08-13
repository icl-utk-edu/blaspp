#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC >
void test_symm_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef scalar_type<TA, TB, TC> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Side side = params.side.value();
    blas::Uplo uplo = params.uplo.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

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

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    lapack_lacpy( "g", Cm, Cn, C, ldc, Cref, ldc );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lansy( "f", uplo2str(uplo), An, A, lda, work );
    real_t Bnorm = lapack_lange( "f", Cm, Cn, B, ldb, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc, work );

    // test error exits
    assert_throw( blas::symm( Layout(0), side,     uplo,     m,  n, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( layout,    Side(0),  uplo,     m,  n, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( layout,    side,     Uplo(0),  m,  n, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( layout,    side,     uplo,    -1,  n, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( layout,    side,     uplo,     m, -1, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::symm( layout, Side::Left,  uplo,     m,  n, alpha, A, m-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( layout, Side::Right, uplo,     m,  n, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::symm( Layout::ColMajor, side, uplo,  m,  n, alpha, A, lda, B, m-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::symm( Layout::RowMajor, side, uplo,  m,  n, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::symm( Layout::ColMajor, side, uplo,  m,  n, alpha, A, lda, B, ldb, beta, C, m-1 ), blas::Error );
    assert_throw( blas::symm( Layout::RowMajor, side, uplo,  m,  n, alpha, A, lda, B, ldb, beta, C, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "side %c, uplo %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B  m=%5lld,  n=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  m=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                side2char(side), uplo2char(uplo),
                (lld) An, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld)  m, (lld)  n, (lld) ldb, (lld) size_B, Bnorm,
                (lld)  m, (lld)  n, (lld) ldc, (lld) size_C, Cnorm );
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
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::symm( layout, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc );
    time = get_wtime() - time;

    double gflop = Gflop < scalar_t >::symm( side, m, n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( Cm, Cn, C, ldc );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        cblas_symm( cblas_layout_const(layout),
                    cblas_side_const(side),
                    cblas_uplo_const(uplo),
                    m, n, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( Cm, Cn, Cref, ldc );
        }

        // check error compared to reference
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, An, alpha, beta, Anorm, Bnorm, Cnorm,
                    Cref, ldc, C, ldc, verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
}

// -----------------------------------------------------------------------------
void test_symm( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_symm_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_symm_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_symm_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_symm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_symm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
