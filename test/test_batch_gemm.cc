#include "test.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "blas.hh"
// -----------------------------------------------------------------------------
template< typename TA, typename TB, typename TC >
void test_batch_gemm_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using namespace blas::batch;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t = blas::real_type< scalar_t >;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Op transA_ = params.transA.value();
    blas::Op transB_ = params.transB.value();
    scalar_t alpha_  = params.alpha.value();
    scalar_t beta_   = params.beta.value();
    int64_t m_       = params.dim.m();
    int64_t n_       = params.dim.n();
    int64_t k_       = params.dim.k();
    size_t  batch   = params.batch.value();
    int64_t align   = params.align.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.gflops.value();
    params.ref_time.value();
    params.ref_gflops.value();

    if ( ! run)
        return;

    // setup
    int64_t Am = (transA_ == Op::NoTrans ? m_ : k_);
    int64_t An = (transA_ == Op::NoTrans ? k_ : m_);
    int64_t Bm = (transB_ == Op::NoTrans ? k_ : n_);
    int64_t Bn = (transB_ == Op::NoTrans ? n_ : k_);
    int64_t Cm = m_;
    int64_t Cn = n_;
    if (layout == Layout::RowMajor) {
        std::swap( Am, An );
        std::swap( Bm, Bn );
        std::swap( Cm, Cn );
    }
    
    int64_t lda_ = roundup( Am, align );
    int64_t ldb_ = roundup( Bm, align );
    int64_t ldc_ = roundup( Cm, align );
    size_t size_A = size_t(lda_)*An;
    size_t size_B = size_t(ldb_)*Bn;
    size_t size_C = size_t(ldc_)*Cn;
    TA* A    = new TA[ batch * size_A ];
    TB* B    = new TB[ batch * size_B ];
    TC* C    = new TC[ batch * size_C ];
    TC* Cref = new TC[ batch * size_C ];

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TB*>    Barray( batch );
    std::vector<TC*>    Carray( batch );
    std::vector<TC*> Crefarray( batch );

    for(size_t i = 0; i < batch; i++){
         Aarray[i]   =  A   + i * size_A;
         Barray[i]   =  B   + i * size_B;
         Carray[i]   =  C   + i * size_C;
        Crefarray[i] = Cref + i * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Op> transA(1, transA_);
    std::vector<blas::Op> transB(1, transB_);
    std::vector<int64_t>  m(1, m_);
    std::vector<int64_t>  n(1, n_);
    std::vector<int64_t>  k(1, k_);
    std::vector<int64_t>  lda(1, lda_);
    std::vector<int64_t>  ldb(1, ldb_);
    std::vector<int64_t>  ldc(1, ldc_);
    std::vector<scalar_t> alpha(1, alpha_);
    std::vector<scalar_t> beta(1, beta_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );
    lapack_larnv( idist, iseed, batch * size_B, B );
    lapack_larnv( idist, iseed, batch * size_C, C );
    lapack_lacpy( "g", Cm, batch * Cn, C, ldc_, Cref, ldc_ );

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    real_t* Cnorm = new real_t[ batch ];
    
    for(size_t i = 0; i < batch; i++){
        Anorm[i] = lapack_lange( "f", Am, An, Aarray[i], lda_, work );
        Bnorm[i] = lapack_lange( "f", Bm, Bn, Barray[i], ldb_, work );
        Cnorm[i] = lapack_lange( "f", Cm, Cn, Carray[i], ldc_, work );
    }

    // decide error checking mode
    info.resize( 0 );

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::batch::gemm( layout, transA, transB, m, n, k, 
                       alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc, 
                       batch, info );
    time = get_wtime() - time;

    double gflop = batch * Gflop < scalar_t >::gemm( m_, n_, k_ );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    if (params.ref.value() == 'y' || params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        for(size_t i = 0; i < batch; i++){
            cblas_gemm( cblas_layout_const(layout),
                        cblas_trans_const(transA_),
                        cblas_trans_const(transB_),
                        m_, n_, k_, alpha_, Aarray[i], lda_, Barray[i], ldb_, beta_, Crefarray[i], ldc_ );
        }
        time = get_wtime() - time;

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // check error compared to reference
        real_t err, error = 0;
        bool ok, okay = true;
        for(size_t i = 0; i < batch; i++){
            check_gemm( Cm, Cn, k_, alpha_, beta_, Anorm[i], Bnorm[i], Cnorm[i],
                        Crefarray[i], ldc_, Carray[i], ldc_, verbose, &err, &ok );
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
}

// -----------------------------------------------------------------------------
void test_batch_gemm( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_batch_gemm_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_batch_gemm_work< float, float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_batch_gemm_work< double, double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_batch_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_batch_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;
    }
}
