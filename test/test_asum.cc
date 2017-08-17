#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack.hh"
#include "flops.hh"
#include "check_gemm.hh"

#include "asum.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_asum_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits<T>::norm_t norm_t;
    typedef long long lld;

    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx.value();
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
    T* x = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );

    if (verbose >= 1) {
        printf( "x n=%5lld, inc=%5lld, size=%5lld\n",
                (lld) n, (lld) incx, (lld) size_x );
    }
    if (verbose >= 2) {
        printf( "x = " ); //print_vector( n, x, abs(incx) );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    norm_t result = blas::asum( n, x, incx );
    time = omp_get_wtime() - time;

    double gflop = gflop_asum( n, x );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 1) {
        printf( "result = %.4e\n", result );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        norm_t ref = cblas_asum( n, x, incx );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 1) {
            printf( "ref    = %.4e\n", ref );
        }

        // error = |ref - result| / |result|
        norm_t error = std::abs( ref - result ) / std::abs( result );
        params.error.value() = error;

        norm_t eps = std::numeric_limits< norm_t >::epsilon();
        norm_t tol = params.tol.value() * eps;
        params.okay.value() = (error < tol);
    }

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_asum( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_asum_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_asum_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_asum_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_asum_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_asum_work< std::complex<double> >( params, run );
            break;
    }
}
