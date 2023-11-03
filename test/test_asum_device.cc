//add this routine by chenjun
#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"

template <typename T>
void test_asum_device_work( Params& params, bool run )
{
    using namespace testsweeper;
    using real_t  = blas::real_type< T >;
    // get & mark input values
    int64_t n       = params.dim.n();
    int64_t incx    = params.incx();
    int64_t verbose = params.verbose();
    int64_t device  = params.device();

    // mark non-standard output values
    params.gflops();
    params.gbytes();
    params.ref_time();
    params.ref_gflops();
    params.ref_gbytes();
    params.runs();
    // adjust header to msec
    params.time.name( "time (ms)" );
    params.ref_time.name( "ref time (ms)" );
    params.ref_time.width( 13 );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // setup
    size_t size_x = (n - 1) * std::abs(incx) + 1;
    T* x = new T[ size_x ];

    blas::Queue queue(device);
    T* dx;
    dx = blas::device_malloc<T>(size_x, queue);

    real_t result_host;
    real_t result_device;
    //result_device = blas::device_malloc<real_t>(1, queue);

    
    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    //init data
    lapack_larnv( idist, iseed, size_x, x );
    //copy data from host to device
    blas::device_copy_vector(n, x, std::abs(incx), dx, std::abs(incx), queue);
    queue.sync();

    // test error exits
    assert_throw( blas::asum( -1, dx, incx , &result_device, queue), blas::Error );
    assert_throw( blas::asum(  n, dx,    0 , &result_device, queue), blas::Error );
    if (verbose >= 1) {
        printf( "\n"
                "x n=%5lld, inc=%5lld, size=%10lld\n",
                llong( n ), llong( incx ), llong( size_x ) );
    }
    if (verbose >= 2) {
        printf( "x = " ); print_vector( n, x, incx );
    }
    // run test
    testsweeper::flush_cache( params.cache() );
    //double time = get_wtime();
    blas::asum( n, dx, incx, &result_device, queue);
    queue.sync();
    //time = get_wtime() - time;
    //copy data from gpu to cpu
    blas::device_copy_vector(1, &result_device, 1, &result_host, 1, queue);
    queue.sync();
    double gflop = blas::Gflop< T >::asum( n );
    double gbyte = blas::Gbyte< T >::asum( n );
    // params.time()   = time * 1000;  // msec
    // params.gflops() = gflop / time;
    // params.gbytes() = gbyte / time;

    if (verbose >= 1) {
        printf( "result = %.4e\n", result_host );
    }
    double time;
    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        real_t ref = cblas_asum( n, x, incx );
        time = get_wtime() - time;

        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;
        params.ref_gbytes() = gbyte / time;

        if (verbose >= 1) {
            printf( "ref    = %.4e\n", ref );
        }

        // relative forward error
        // note: using sqrt(n) here gives failures
        real_t error = std::abs( (ref - result_host) / (n * ref) );

        // complex needs extra factor; see Higham, 2002, sec. 3.6.
        if (blas::is_complex<T>::value) {
            error /= 2*sqrt(2);
        }

        real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
        params.error() = error;
        params.okay() = (error < u);
    }

    int runs = params.runs();
    double stime;
    double all_time=0.0f;
    for(int i = 0; i < runs; i++){
        testsweeper::flush_cache( params.cache() );
        stime = get_wtime();
        blas::asum( n, dx, incx, &result_device, queue);
        queue.sync();
        all_time += (get_wtime() - stime);
    }
    all_time/=(double)runs;
    params.time()   = all_time * 1000;  // msec
    params.gflops() = gflop / all_time;
    params.gbytes() = gbyte / all_time;

    delete[] x;
    blas::device_free(dx, queue);
}



// -----------------------------------------------------------------------------
void test_asum_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_asum_device_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_asum_device_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_asum_device_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_asum_device_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}