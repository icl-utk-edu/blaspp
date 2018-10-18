#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC >
void test_syr2k_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef scalar_type<TA, TC> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Op trans  = params.trans();
    blas::Uplo uplo = params.uplo();
    scalar_t alpha  = params.alpha();
    scalar_t beta   = params.beta();
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
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Am, align );
    int64_t ldc = roundup(  n, align );
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
    real_t Cnorm = lapack_lansy( "f", uplo2str(uplo), n, C, ldc, work );

    // test error exits
    assert_throw( blas::syr2k( Layout(0), uplo,    trans,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( layout,    Uplo(0), trans,  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    Op(0),  n,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    trans, -1,  k, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( layout,    uplo,    trans,  n, -1, alpha, A, lda, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, A, k-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, A, n-1, B, ldb, beta, C, ldc ), blas::Error );

    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::NoTrans,   n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::Trans,     n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::ColMajor, uplo, Op::ConjTrans, n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::NoTrans,   n, k, alpha, A, lda, B, k-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::Trans,     n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );
    assert_throw( blas::syr2k( Layout::RowMajor, uplo, Op::ConjTrans, n, k, alpha, A, lda, B, n-1, beta, C, ldc ), blas::Error );

    assert_throw( blas::syr2k( layout,    uplo,    trans,  n,  k, alpha, A, lda, B, ldb, beta, C, n-1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "uplo %c, trans %c\n"
                "A An=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm %.2e\n"
                "B Bn=%5lld, Bn=%5lld, ldb=%5lld, size=%10lld, norm %.2e\n"
                "C  n=%5lld,  n=%5lld, ldc=%5lld, size=%10lld, norm %.2e\n",
                uplo2char(uplo), op2char(trans),
                (lld) Am, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld) Am, (lld) An, (lld) ldb, (lld) size_B, Bnorm,
                (lld)  n, (lld)  n, (lld) ldc, (lld) size_C, Cnorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( Am, An, A, lda );
        printf( "B = "    ); print_matrix( Am, An, B, ldb );
        printf( "C = "    ); print_matrix(  n,  n, C, ldc );
    }

    // run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    blas::syr2k( layout, uplo, trans, n, k,
                 alpha, A, lda, B, ldb, beta, C, ldc );
    time = get_wtime() - time;

    double gflop = Gflop < scalar_t >::syr2k( n, k );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        printf( "C2 = " ); print_matrix( n, n, C, ldc );
    }

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        cblas_syr2k( cblas_layout_const(layout),
                     cblas_uplo_const(uplo),
                     cblas_trans_const(trans),
                     n, k, alpha, A, lda, B, ldb, beta, Cref, ldc );
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            printf( "Cref = " ); print_matrix( n, n, Cref, ldc );
        }

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

// -----------------------------------------------------------------------------
void test_syr2k( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            //test_syr2k_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_syr2k_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr2k_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_syr2k_work< std::complex<float>, std::complex<float>,
                             std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_syr2k_work< std::complex<double>, std::complex<double>,
                             std::complex<double> >( params, run );
            break;
    }
}
