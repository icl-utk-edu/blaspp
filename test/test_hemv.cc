#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "hemv.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TX, typename TY >
void test_hemv_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits3< TA, TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Uplo uplo = params.uplo.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t incy    = params.incy.value();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.gbytes.value();
    params.ref_time.value();
    params.ref_gflops.value();
    params.ref_gbytes.value();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

    // setup
    int64_t lda = roundup( n, align );
    size_t size_A = size_t(lda)*n;
    size_t size_x = (n - 1) * abs(incx) + 1;
    size_t size_y = (n - 1) * abs(incy) + 1;
    TA* A    = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    cblas_copy( n, y, incy, yref, incy );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lanhe( "f", uplo2str(uplo), n, A, lda, work );
    real_t Xnorm = cblas_nrm2( n, x, abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, abs(incy) );

    // test error exits
    assert_throw( blas::hemv( Layout(0), uplo,     n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::hemv( layout,    Uplo(0),  n, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::hemv( layout,    uplo,    -1, alpha, A, lda, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::hemv( layout,    uplo,     n, alpha, A, n-1, x, incx, beta, y, incy ), blas::Error );
    assert_throw( blas::hemv( layout,    uplo,     n, alpha, A, lda, x,    0, beta, y, incy ), blas::Error );
    assert_throw( blas::hemv( layout,    uplo,     n, alpha, A, lda, x, incx, beta, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A n=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm=%.2e\n",
                (lld) n, (lld) lda,  (lld) size_A, Anorm,
                (lld) n, (lld) incx, (lld) size_x, Xnorm,
                (lld) n, (lld) incy, (lld) size_y, Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei; beta = %.4e + %.4ei;\n",
                real(alpha), imag(alpha),
                real(beta),  imag(beta) );
        printf( "A = "    ); print_matrix( n, n, A, lda );
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    blas::hemv( layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy );
    time = omp_get_wtime() - time;

    double gflop = Gflop < scalar_t >::hemv( n );
    double gbyte = Gbyte < scalar_t >::hemv( n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_hemv( cblas_layout_const(layout), cblas_uplo_const(uplo), n,
                    alpha, A, lda, x, incx, beta, yref, incy );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // check error compared to reference
        // treat y as 1 x leny matrix with ld = incy; k = lenx is reduction dimension
        real_t error;
        bool okay;
        check_gemm( 1, n, n, alpha, beta, Anorm, Xnorm, Ynorm,
                    yref, abs(incy), y, abs(incy), verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] x;
    delete[] y;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_hemv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_hemv_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_hemv_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_hemv_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_hemv_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_hemv_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
