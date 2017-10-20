#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack.hh"
#include "flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "dot.hh"

// -----------------------------------------------------------------------------
template< typename TX, typename TY >
void test_dotu_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits2< TX, TY >::scalar_t scalar_t;
    typedef typename traits< scalar_t >::real_t real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
    int64_t incy    = params.incy.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    // adjust header names
    params.time.name( "SLATE\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

    // setup
    size_t size_x = (n - 1) * abs(incx) + 1;
    size_t size_y = (n - 1) * abs(incy) + 1;
    TX* x = new TX[ size_x ];
    TY* y = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );

    // norms for error check
    real_t Xnorm = cblas_nrm2( n, x, abs(incx) );
    real_t Ynorm = cblas_nrm2( n, y, abs(incy) );

    // test error exits
    assert_throw( blas::dotu( -1, x, incx, y, incy ), blas::Error );
    assert_throw( blas::dotu(  n, x,    0, y, incy ), blas::Error );
    assert_throw( blas::dotu(  n, x, incx, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n"
                "y n=%5lld, inc=%5lld, size=%10lld, norm %.2e\n",
                (lld) n, (lld) incx, (lld) size_x, Xnorm,
                (lld) n, (lld) incy, (lld) size_y, Ynorm );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
        printf( "y = " ); print_vector( n, y, incy );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    scalar_t result = blas::dotu( n, x, incx, y, incy );
    time = omp_get_wtime() - time;

    double gflop = gflop_dot( n, x );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "dotu = %.4e + %.4ei\n", real(result), imag(result) );
    }

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        scalar_t ref = cblas_dotu( n, x, incx, y, incy );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "ref = %.4e + %.4ei\n", real(ref), imag(ref) );
        }

        // check error compared to reference
        // treat result as 1 x 1 matrix; k = n is reduction dimension
        // alpha=1, beta=0, Cnorm=0
        real_t error;
        int64_t okay;
        check_gemm( 1, 1, n, scalar_t(1), scalar_t(0), Xnorm, Ynorm, real_t(0),
                    &ref, 1, &result, 1, &error, &okay );
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_dotu( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_dotu_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_dotu_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_dotu_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_dotu_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_dotu_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
