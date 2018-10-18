#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template< typename TX, typename TY >
void test_copy_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef scalar_type<TX, TY> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t incy    = params.incy();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if ( ! run)
        return;

    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    size_t size_y = (n - 1) * std::abs(incy) + 1;
    TX* x    = new TX[ size_x ];
    TY* y    = new TY[ size_y ];
    TY* yref = new TY[ size_y ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    lapack_larnv( idist, iseed, size_y, y );
    lapack_larnv( idist, iseed, size_y, yref );

    // test error exits
    assert_throw( blas::copy( -1, x, incx, y, incy ), blas::Error );
    assert_throw( blas::copy(  n, x,    0, y, incy ), blas::Error );
    assert_throw( blas::copy(  n, x, incx, y,    0 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n"
                "y n=%5lld, inc=%5lld, size=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x,
                (lld) n, (lld) incy, (lld) size_y );
    }
    if (verbose >= 2) {
        printf( "x    = " ); print_vector( n, x, incx );
        printf( "y    = " ); print_vector( n, y, incy );
    }

    // run test
    libtest::flush_cache( params.cache() );
    double time = get_wtime();
    blas::copy( n, x, incx, y, incy );
    time = get_wtime() - time;

    double gflop = Gflop < scalar_t >::copy( n );
    double gbyte = Gbyte < scalar_t >::copy( n );
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;
    params.gbytes() = gbyte / time;

    if (verbose >= 2) {
        printf( "y2   = " ); print_vector( n, y, incy );
    }

    if (params.check() == 'y') {
        // run reference
        libtest::flush_cache( params.cache() );
        time = get_wtime();
        cblas_copy( n, x, incx, yref, incy );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 2) {
            printf( "yref = " ); print_vector( n, yref, incy );
        }

        // error = ||yref - y||
        cblas_axpy( n, -1.0, y, incy, yref, incy );
        real_t error = cblas_nrm2( n, yref, std::abs(incy) );
        params.error() = error;

        // copy must be exact!
        params.okay() = (error == 0);
    }

    delete[] x;
    delete[] y;
    delete[] yref;
}

// -----------------------------------------------------------------------------
void test_copy( Params& params, bool run )
{
    switch (params.datatype()) {
        case libtest::DataType::Integer:
            //test_copy_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_copy_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_copy_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_copy_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_copy_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
