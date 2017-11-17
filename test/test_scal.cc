#include <omp.h>

#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"

#include "scal.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_scal_work( Params& params, bool run )
{
    using namespace blas;
    typedef typename traits<T>::real_t real_t;
    typedef long long lld;

    // get & mark input values
    T alpha         = params.alpha.value();
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
    T* x    = new T[ size_x ];
    T* xref = new T[ size_x ];

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_x, x );
    cblas_copy( n, x, incx, xref, incx );

    // test error exits
    assert_throw( blas::scal( -1, alpha, x, incx ), blas::Error );
    assert_throw( blas::scal(  n, alpha, x,    0 ), blas::Error );
    assert_throw( blas::scal(  n, alpha, x,   -1 ), blas::Error );

    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                (lld) n, (lld) incx, (lld) size_x );
    }
    if (verbose >= 2) {
        printf( "alpha = %.4e + %.4ei;\n",
                real(alpha), imag(alpha) );
        printf( "x    = " ); print_vector( n, x, incx );
    }

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = omp_get_wtime();
    blas::scal( n, alpha, x, incx );
    time = omp_get_wtime() - time;

    double gflop = gflop_scal( n, x );
    params.time.value()   = time * 1000;  // msec
    params.gflops.value() = gflop / time;

    if (verbose >= 2) {
        printf( "x2   = " ); print_vector( n, x, incx );
    }

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = omp_get_wtime();
        cblas_scal( n, alpha, xref, incx );
        time = omp_get_wtime() - time;

        params.ref_time.value()   = time * 1000;  // msec
        params.ref_gflops.value() = gflop / time;

        if (verbose >= 2) {
            printf( "xref = " ); print_vector( n, xref, incx );
        }

        // error = ||xref - x|| / ||x|| ... todo
        cblas_axpy( n, -1.0, x, incx, xref, incx );
        real_t error = cblas_nrm2( n, xref, abs(incx) );
        real_t xnorm = cblas_nrm2( n, x,    abs(incx) );
        error /= xnorm;
        params.error.value() = error;

        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol.value() * eps;
        params.okay.value() = (error < tol);
    }

    delete[] x;
    delete[] xref;
}

// -----------------------------------------------------------------------------
void test_scal( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_scal_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_scal_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_scal_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_scal_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_scal_work< std::complex<double> >( params, run );
            break;
    }
}
