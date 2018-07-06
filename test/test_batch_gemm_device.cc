#include "test_device.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "device_blas.hh"
// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC >
void test_device_batch_gemm_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using namespace blas::batch;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op transA = params.transA.value();
    blas::Op transB = params.transB.value();
    scalar_t alpha  = params.alpha.value();
    scalar_t beta   = params.beta.value();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t k       = params.dim.k();
    int64_t batch   = params.batch.value();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if ( ! run)
        return;

    // setup
    int64_t Am = (transA == Op::NoTrans ? m : k);
    int64_t An = (transA == Op::NoTrans ? k : m);
    int64_t Bm = (transB == Op::NoTrans ? k : n);
    int64_t Bn = (transB == Op::NoTrans ? n : k);
    int64_t Cm = m;
    int64_t Cn = n;
    
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    int64_t ldc = roundup( Cm, align );
    size_t size_A = size_t(lda)*An;
    size_t size_B = size_t(ldb)*Bn;
    size_t size_C = size_t(ldc)*Cn;
    TA* A    = new TA[ batch * size_A ];
    TB* B    = new TB[ batch * size_B ];
    TC* C    = new TC[ batch * size_C ];
    TC* Cref = new TC[ batch * size_C ];

    // device specifics 
    blas::Queue queue(0, batch);
    TA* dA; 
    TB* dB; 
    TC* dC;

    dA = blas::device_malloc<TA>( batch * size_A );
    dB = blas::device_malloc<TB>( batch * size_B );
    dC = blas::device_malloc<TC>( batch * size_C );

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TB*>    Barray( batch );
    std::vector<TC*>    Carray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TB*>   dBarray( batch );
    std::vector<TC*>   dCarray( batch );
    std::vector<TC*> Crefarray( batch );

    for(int64_t i = 0; i < batch; i++){
         Aarray[i]   =  A   + i * size_A;
         Barray[i]   =  B   + i * size_B;
         Carray[i]   =  C   + i * size_C;
        dAarray[i]   = dA   + i * size_A;
        dBarray[i]   = dB   + i * size_B;
        dCarray[i]   = dC   + i * size_C;
        Crefarray[i] = Cref + i * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Op> vtransA(1, transA);
    std::vector<blas::Op> vtransB(1, transB);
    std::vector<int64_t> vm(1, m);
    std::vector<int64_t> vn(1, n);
    std::vector<int64_t> vk(1, k);
    std::vector<int64_t> vldda(1, lda);
    std::vector<int64_t> vlddb(1, ldb);
    std::vector<int64_t> vlddc(1, ldc);
    std::vector<scalar_t> valpha(1, alpha);
    std::vector<scalar_t> vbeta(1, beta);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_B, B );
    lapack_larnv( idist, iseed, batch * size_C, C );
    lapack_lacpy( "g", Cm, batch * Cn, C, ldc, Cref, ldc );

    blas::device_setmatrix(Am, batch * An, A, lda, dA, lda, queue);
    blas::device_setmatrix(Bm, batch * Bn, B, ldb, dB, ldb, queue);
    blas::device_setmatrix(Cm, batch * Cn, C, ldc, dC, ldc, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    real_t* Cnorm = new real_t[ batch ];
    
    for(int64_t i = 0; i < batch; i++){
        Anorm[i] = lapack_lange( "f", Am, An, Aarray[i], lda, work );
        Bnorm[i] = lapack_lange( "f", Bm, Bn, Barray[i], ldb, work );
        Cnorm[i] = lapack_lange( "f", Cm, Cn, Carray[i], ldc, work );
    }

    // decide error checking mode
    info.resize( 0 );
    // run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::batch::gemm( vtransA, vtransB, vm, vn, vk, 
                       valpha, dAarray, vldda, dBarray, vlddb, vbeta, dCarray, vlddc, 
                       batch, info, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = batch * Gflop < scalar_t >::gemm( m, n, k );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;
    blas::device_getmatrix(Cm, batch * Cn, dC, ldc, C, ldc, queue);
    queue.sync();

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        for(int64_t i = 0; i < batch; i++){
            cblas_gemm( cblas_layout_const(layout),
                        cblas_trans_const(transA),
                        cblas_trans_const(transB),
                        m, n, k, alpha, Aarray[i], lda, Barray[i], ldb, beta, Crefarray[i], ldc );
        }
        time = get_wtime() - time;

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // check error compared to reference
        real_t err, error = 0;
        bool ok, okay = true;
        for(int64_t i = 0; i < batch; i++){
            check_gemm( Cm, Cn, k, alpha, beta, Anorm[i], Bnorm[i], Cnorm[i],
                        Crefarray[i], ldc, Carray[i], ldc, verbose, &err, &ok );
            error = max(error, err);
            okay &= ok;
        }
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] C;
    delete[] Cref;
    delete[] Anorm;
    delete[] Bnorm;
    delete[] Cnorm;

    blas::device_free( dA );
    blas::device_free( dB );
    blas::device_free( dC );
}

// -----------------------------------------------------------------------------
void test_batch_gemm_device( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_device_batch_gemm_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_device_batch_gemm_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_device_batch_gemm_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_device_batch_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_device_batch_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
