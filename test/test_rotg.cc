#include "test.hh"
#include "cblas.hh"
#include "lapack_wrappers.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotg_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::real;
    using blas::imag;
    typedef blas::real_type<T> real_t;

    // get & mark input values
    int64_t n = params.dim.n();

    // mark non-standard output values
    params.ref_time();

    // adjust header to msec
    params.time.name( "BLAS++\ntime (ms)" );
    params.ref_time.name( "Ref.\ntime (ms)" );

    if (! run)
        return;

    // setup
    std::vector<T> a( n ), aref( n );
    std::vector<T> b( n ), bref( n );
    std::vector<T> s( n ), sref( n );
    std::vector< blas::real_type<T> > c( n ), cref( n );

    int64_t idist = 3;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, n, &a[0] );
    lapack_larnv( idist, iseed, n, &b[0] );
    aref = a;
    bref = b;

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    for (int64_t i = 0; i < n; ++i) {
        blas::rotg( &a[i], &b[i], &c[i], &s[i] );
    }
    time = get_wtime() - time;
    params.time() = time * 1000;  // msec

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (int64_t i = 0; i < n; ++i) {
            cblas_rotg( &aref[i], &bref[i], &cref[i], &sref[i] );
        }
        time = get_wtime() - time;
        params.ref_time() = time * 1000;  // msec

        // get max error of all outputs
        cblas_axpy( n, -1.0, &a[0], 1, &aref[0], 1 );
        cblas_axpy( n, -1.0, &b[0], 1, &bref[0], 1 );
        cblas_axpy( n, -1.0, &c[0], 1, &cref[0], 1 );
        cblas_axpy( n, -1.0, &s[0], 1, &sref[0], 1 );

        int64_t ia = cblas_iamax( n, &aref[0], 1 );
        int64_t ib = cblas_iamax( n, &bref[0], 1 );
        int64_t ic = cblas_iamax( n, &cref[0], 1 );
        int64_t is = cblas_iamax( n, &sref[0], 1 );

        real_t error = blas::max(
            std::abs( aref[ ia ] ),
            std::abs( bref[ ib ] ),
            std::abs( cref[ ic ] ),
            std::abs( sref[ is ] )
        );

        // error is normally 0, but allow for some rounding just in case.
        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < 10*u);
    }
}

// -----------------------------------------------------------------------------
void test_rotg( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_rotg_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_rotg_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_rotg_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_rotg_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
