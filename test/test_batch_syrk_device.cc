#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "device_blas.hh"
// -----------------------------------------------------------------------------
template< typename TA, typename TC >
void test_batch_syrk_device_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    typedef scalar_type<TA, TC> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op trans_     = params.trans.value();
    blas::Uplo uplo_    = params.uplo.value();
    scalar_t alpha_     = params.alpha.value();
    scalar_t beta_      = params.beta.value();
    int64_t n_          = params.dim.n();
    int64_t k_          = params.dim.k();
    size_t  batch      = params.batch.value();
    int64_t align      = params.align.value();
    int64_t verbose    = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if ( ! run)
        return;

    // setup
    int64_t Am = (trans_ == Op::NoTrans ? n_ : k_);
    int64_t An = (trans_ == Op::NoTrans ? k_ : n_);
    if (layout == Layout::RowMajor)
        std::swap( Am, An );
    int64_t lda_ = roundup( Am, align );
    int64_t ldc_ = roundup( n_, align );
    size_t size_A = size_t(lda_)*An;
    size_t size_C = size_t(ldc_)*n_;
    TA* A    = new TA[ batch * size_A ];
    TC* C    = new TC[ batch * size_C ];
    TC* Cref = new TC[ batch * size_C ];

    // device specifics 
    blas::Queue queue(0, batch);
    TA* dA = blas::device_malloc<TA>( batch * size_A );
    TC* dC = blas::device_malloc<TC>( batch * size_C );

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TC*>    Carray( batch );
    std::vector<TC*> Crefarray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TC*>   dCarray( batch );

    for(size_t i = 0; i < batch; i++){
         Aarray[i]   =  A   + i * size_A;
         Carray[i]   =  C   + i * size_C;
        Crefarray[i] = Cref + i * size_C;
        dAarray[i]   = dA   + i * size_A;
        dCarray[i]   = dC   + i * size_C;        
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Uplo> uplo(1, uplo_);
    std::vector<blas::Op>   trans(1, trans_);
    std::vector<int64_t>    n(1, n_);
    std::vector<int64_t>    k(1, k_);
    std::vector<int64_t>    lda(1, lda_);
    std::vector<int64_t>    ldc(1, ldc_);
    std::vector<scalar_t>   alpha(1, alpha_);
    std::vector<scalar_t>   beta(1, beta_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_C, C );
    lapack_lacpy( "g", n_, batch * n_, C, ldc_, Cref, ldc_ );

    blas::device_setmatrix(Am, batch * An, A, lda_, dA, lda_, queue);
    blas::device_setmatrix(n_, batch * n_, C, ldc_, dC, ldc_, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Cnorm = new real_t[ batch ];

    for(size_t s = 0; s < batch; s++){
        Anorm[s] = lapack_lange( "f", Am, An, Aarray[s], lda_, work );
        Cnorm[s] = lapack_lansy( "f", uplo2str(uplo_), n_, Carray[s], ldc_, work );
    }

    // decide error checking mode
    info.resize( 0 );

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::batch::syrk( uplo, trans, n, k, alpha, dAarray, lda, beta, dCarray, ldc, 
                       batch, info, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = batch * Gflop < scalar_t >::syrk( n_, k_ );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;
    blas::device_getmatrix(n_, batch * n_, dC, ldc_, C, ldc_, queue);
    queue.sync();

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        for(size_t s = 0; s < batch; s++){
            cblas_syrk( cblas_layout_const(layout),
                        cblas_uplo_const(uplo_),
                        cblas_trans_const(trans_),
                        n_, k_, alpha_, Aarray[s], lda_, beta_, Crefarray[s], ldc_ );
        }
        time = get_wtime() - time;

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // check error compared to reference
        real_t err, error = 0;
        bool ok, okay = true;
        for(size_t s = 0; s < batch; s++){
            check_herk( uplo_, n_, k_, alpha_, beta_, Anorm[s], Anorm[s], Cnorm[s],
                        Crefarray[s], ldc_, Carray[s], ldc_, verbose, &err, &ok );

            error = max( error, err );
            okay &= ok;
        }

        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] C;
    delete[] Cref;
    delete[] Anorm;
    delete[] Cnorm;

    blas::device_free( dA );
    blas::device_free( dC );
}

// -----------------------------------------------------------------------------
void test_batch_syrk_device( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_batch_syrk_device_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_batch_syrk_device_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_batch_syrk_device_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_batch_syrk_device_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_batch_syrk_device_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
