#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "geru.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TX, typename TY >
void test_geru_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits3< TA, TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    scalar_t alpha  = params.alpha.value();
    int64_t m       = params.dim.m();
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
    int64_t Am = (layout == Layout::ColMajor ? m : n);
    int64_t An = (layout == Layout::ColMajor ? n : m);
    int64_t lda = roundup( Am, align );
    size_t size_A = size_t(lda)*An;
    size_t size_x = (m - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    TA* A    = new TA[ size_A ];
    TA* Aref = new TA[ size_A ];
    TX* x    = new TX[ size_x ];
    TX* y    = new TX[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    lapack_lacpy( "g", Am, An, A, lda, Aref, lda );

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda, work );
    real_t Xnorm = cblas_nrm2( m, x, std::abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, std::abs(incy) );

    // test error exits
    assert_throw( blas::geru( Layout(0),  m,  n, alpha, A, lda, x, incx, y, incy ), blas::Error );
    assert_throw( blas::geru( layout,    -1,  n, alpha, A, lda, x, incx, y, incy ), blas::Error );
    assert_throw( blas::geru( layout,     m, -1, alpha, A, lda, x, incx, y, incy ), blas::Error );

    assert_throw( blas::geru( Layout::ColMajor,  m,  n, alpha, A, m-1, x, incx, y, incy ), blas::Error );
    assert_throw( blas::geru( Layout::RowMajor,  m,  n, alpha, A, n-1, x, incx, y, incy ), blas::Error );

    assert_throw( blas::geru( layout,     m,  n, alpha, A, lda, x, 0,    y, incy ), blas::Error );
    assert_throw( blas::geru( layout,     m,  n, alpha, A, lda, x, incx, y, 0    ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "A Am=%5lld, An=%5lld, lda=%5lld, size=%10lld, norm=%.2e\n"
                "x Xm=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n"
                "y Ym=%5lld, inc=%5lld,           size=%10lld, norm=%.2e\n",
                (lld) Am, (lld) An, (lld) lda, (lld) size_A, Anorm,
                (lld)  m, (lld) incx,          (lld) size_x, Xnorm,
                (lld)  n, (lld) incy,          (lld) size_y, Ynorm );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "A = " ); print_matrix( Am, An, A, lda );
        printf( "x = " ); print_vector( m, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    blas::geru( layout, m, n, alpha, x, incx, y, incy, A, lda );
    time = omp_get_wtime() - time;

    double gflop = Gflop < scalar_t >::ger( m, n );
    double gbyte = Gbyte < scalar_t >::ger( m, n );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;
    params.gbytes.value() = gbyte / time;

    if (verbose >= 2) {
        printf( "A2 = " ); print_matrix( Am, An, A, lda );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_geru( cblas_layout_const(layout), m, n, alpha, x, incx, y, incy, Aref, lda );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;
        params.ref_gbytes.value() = gbyte / time;

        if (verbose >= 2) {
            printf( "Aref = " ); print_matrix( Am, An, Aref, lda );
        }

        // check error compared to reference
        // beta = 1
        real_t error;
        bool okay;
        check_gemm( Am, An, 1, alpha, scalar_t(1), Xnorm, Ynorm, Anorm,
                    Aref, lda, A, lda, verbose, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] Aref;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_geru( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_geru_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_geru_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_geru_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_geru_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_geru_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
