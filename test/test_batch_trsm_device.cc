#include "test_device.hh"
#include "cblas.hh"
#include "lapack_tmp.hh"
#include "blas_flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

#include "batch_trsm_device.hh"

// -----------------------------------------------------------------------------
template< typename TA, typename TB >
void test_device_batch_trsm_work( Params& params, bool run )
{
    using namespace libtest;
    using namespace blas;
    using namespace blas::batch;
    typedef scalar_type<TA, TB> scalar_t;
    typedef real_type<scalar_t> real_t;
    typedef long long lld;

    // get & mark input values
    blas::Layout layout = params.layout.value();
    blas::Side side = params.side.value();
    blas::Uplo uplo = params.uplo.value();
    blas::Op trans  = params.trans.value();
    blas::Diag diag = params.diag.value();
    scalar_t alpha  = params.alpha.value();
    int64_t m       = params.dim.m();
    int64_t n       = params.dim.n();
    int64_t align   = params.align.value();
    int64_t batch   = params.batch.value();
    int64_t verbose = params.verbose.value();

    // mark non-standard output values
    params.ref_time.value();
    params.ref_gflops.value();

    if (! run)
        return;

    // ----------
    // setup
    int64_t Am = (side == Side::Left ? m : n);
    int64_t Bm = m;
    int64_t Bn = n;
    int64_t lda = roundup( Am, align );
    int64_t ldb = roundup( Bm, align );
    size_t size_A = size_t(lda)*Am;
    size_t size_B = size_t(ldb)*Bn;
    TA* A    = new TA[ batch * size_A ];
    TB* B    = new TB[ batch * size_B ];
    TB* Bref = new TB[ batch * size_B ];

    // device specifics 
    blas::Queue queue(0, batch);
    TA* dA; 
    TB* dB; 

    dA = blas::device_malloc<TA>( batch * size_A );
    dB = blas::device_malloc<TB>( batch * size_B );

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TB*>    Barray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TB*>   dBarray( batch );
    std::vector<TB*> Brefarray( batch );

    for(int i = 0; i < batch; i++){
         Aarray[i]   =  A   + i * size_A;
         Barray[i]   =  B   + i * size_B;
        dAarray[i]   = dA   + i * size_A;
        dBarray[i]   = dB   + i * size_B;
        Brefarray[i] = Bref + i * size_B;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<Side> vside(1, side);
    std::vector<Uplo> vuplo(1, uplo);
    std::vector<Op>   vtrans(1, trans);
    std::vector<Diag> vdiag(1, diag);

    std::vector<int64_t> vm(1, m);
    std::vector<int64_t> vn(1, n);
    std::vector<int64_t> vldda(1, lda);
    std::vector<int64_t> vlddb(1, ldb);
    std::vector<scalar_t> valpha(1, alpha);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );  // TODO: generate
    lapack_larnv( idist, iseed, batch * size_B, B );  // TODO
    lapack_lacpy( "g", Bm, batch * Bn, B, ldb, Bref, ldb );

    // set unused data to nan
    /*if (uplo == Uplo::Lower) {
        for (int j = 0; j < Am; ++j)
            for (int i = 0; i < j; ++i)  // upper
                A[ i + j*lda ] = nan("");
    }
    else {
        for (int j = 0; j < Am; ++j)
            for (int i = j+1; i < Am; ++i)  // lower
                A[ i + j*lda ] = nan("");
    }*/

    // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
    // If diag == Unit, the diagonal is replaced; this is still well-conditioned.
    for(int s = 0; s < batch; s++){
        TA* pA = Aarray[s];
        // First, brute force positive definiteness.
        for (int i = 0; i < Am; ++i) {
            pA[ i + i*lda ] += Am;
        }
        blas_int potrf_info = 0;
        lapack_potrf( uplo2str(uplo), Am, pA, lda, &potrf_info );
        assert( potrf_info == 0 );
    }
    blas::device_setmatrix(Am, batch * Am, A, lda, dA, lda, queue);
    blas::device_setmatrix(Bm, batch * Bn, B, ldb, dB, ldb, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    for(int s = 0; s < batch; s++){
        Anorm[ s ] = lapack_lantr( "f", uplo2str(uplo), diag2str(diag), Am, Am, Aarray[s], lda, work );
        Bnorm[ s ] = lapack_lange( "f", Bm, Bn, Barray[s], ldb, work );
    }

    // test error exits
    /*
    assert_throw( blas::trsm( Layout(0), side,    uplo,    trans, diag,     m,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    Side(0), uplo,    trans, diag,     m,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    side,    Uplo(0), trans, diag,     m,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    side,    uplo,    Op(0), diag,     m,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    side,    uplo,    trans, Diag(0),  m,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    side,    uplo,    trans, diag,    -1,  n, alpha, A, lda, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout,    side,    uplo,    trans, diag,     m, -1, alpha, A, lda, B, ldb ), blas::Error );

    assert_throw( blas::trsm( layout, Side::Left,  uplo,   trans, diag,     m,  n, alpha, A, m-1, B, ldb ), blas::Error );
    assert_throw( blas::trsm( layout, Side::Right, uplo,   trans, diag,     m,  n, alpha, A, n-1, B, ldb ), blas::Error );

    assert_throw( blas::trsm( Layout::ColMajor, side, uplo, trans, diag,    m,  n, alpha, A, lda, B, m-1 ), blas::Error );
    assert_throw( blas::trsm( Layout::RowMajor, side, uplo, trans, diag,    m,  n, alpha, A, lda, B, n-1 ), blas::Error );
    */

    // decide error checking mode
    info.resize( 0 );

    // run test
    libtest::flush_cache( params.cache.value() );
    double time = get_wtime();
    blas::batch::trsm( vside, vuplo, vtrans, vdiag, vm, vn, valpha, dAarray, vldda, dBarray, vlddb, batch, info, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = batch * Gflop < scalar_t >::trsm( side, m, n );
    params.time.value()   = time;
    params.gflops.value() = gflop / time;

    blas::device_getmatrix(Bm, batch * Bn, dB, ldb, B, ldb, queue);
    queue.sync();

    if (params.check.value() == 'y') {
        // run reference
        libtest::flush_cache( params.cache.value() );
        time = get_wtime();
        for(int i = 0; i < batch; i++){
            cblas_trsm( cblas_layout_const(layout),
                        cblas_side_const(side),
                        cblas_uplo_const(uplo),
                        cblas_trans_const(trans),
                        cblas_diag_const(diag),
                        m, n, alpha, Aarray[i], lda, Brefarray[i], ldb );
        }
        time = get_wtime() - time;

        params.ref_time.value()   = time;
        params.ref_gflops.value() = gflop / time;

        // check error compared to reference
        // Am is reduction dimension
        // beta = 0, Cnorm = 0 (initial).
        real_t err, error = 0;
        bool ok, okay = true;
        for(int i = 0; i < batch; i++){
            check_gemm( Bm, Bn, Am, alpha, scalar_t(0), Anorm[i], Bnorm[i], real_t(0),
                        Brefarray[i], ldb, Barray[i], ldb, verbose, &err, &ok );
            error = max(error, err);
            okay &= ok;
        }
        params.error.value() = error;
        params.okay.value() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] Bref;
    delete[] Anorm;
    delete[] Bnorm;

    blas::device_free( dA );
    blas::device_free( dB );
}

// -----------------------------------------------------------------------------
void test_batch_trsm_device( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_device_batch_trsm_work< int64_t >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_device_batch_trsm_work< float, float >( params, run );
            break;

        case libtest::DataType::Double:
            test_device_batch_trsm_work< double, double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_device_batch_trsm_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_device_batch_trsm_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;
    }
}
