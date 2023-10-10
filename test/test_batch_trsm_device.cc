// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "cblas_wrappers.hh"
#include "lapack_wrappers.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TB>
void test_device_batch_trsm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas::batch;
    using blas::Uplo;
    using blas::Side;
    using blas::Op;
    using blas::Layout;
    using blas::Diag;
    using scalar_t = blas::scalar_type< TA, TB >;
    using real_t   = blas::real_type< scalar_t >;
    using std::swap;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side_    = params.side();
    blas::Uplo uplo_    = params.uplo();
    blas::Op trans_     = params.trans();
    blas::Diag diag_    = params.diag();
    scalar_t alpha_     = params.alpha();
    int64_t m_          = params.dim.m();
    int64_t n_          = params.dim.n();
    size_t batch        = params.batch();
    int64_t device      = params.device();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // ----------
    // setup
    int64_t Am    = (side_ == Side::Left ? m_ : n_);
    int64_t Bm    = m_;
    int64_t Bn    = n_;
    if (layout == Layout::RowMajor)
        swap( Bm, Bn );
    int64_t lda_  = roundup( Am, align );
    int64_t ldb_  = roundup( Bm, align );
    size_t size_A = size_t(lda_)*Am;
    size_t size_B = size_t(ldb_)*Bn;
    TA* A         = new TA[ batch * size_A ];
    TB* B         = new TB[ batch * size_B ];
    TB* Bref      = new TB[ batch * size_B ];

    // device specifics
    blas::Queue queue( device );
    TA* dA;
    TB* dB;

    dA = blas::device_malloc<TA>( batch * size_A, queue );
    dB = blas::device_malloc<TB>( batch * size_B, queue );

    // pointer arrays
    std::vector<TA*>    Aarray( batch );
    std::vector<TB*>    Barray( batch );
    std::vector<TA*>   dAarray( batch );
    std::vector<TB*>   dBarray( batch );
    std::vector<TB*> Brefarray( batch );

    for (size_t i = 0; i < batch; ++i) {
         Aarray[i]   =  A   + i * size_A;
         Barray[i]   =  B   + i * size_B;
        dAarray[i]   = dA   + i * size_A;
        dBarray[i]   = dB   + i * size_B;
        Brefarray[i] = Bref + i * size_B;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<Side>     side(1, side_);
    std::vector<Uplo>     uplo(1, uplo_);
    std::vector<Op>       trans(1, trans_);
    std::vector<Diag>     diag(1, diag_);
    std::vector<int64_t>  m(1, m_);
    std::vector<int64_t>  n(1, n_);
    std::vector<int64_t>  ldda(1, lda_);
    std::vector<int64_t>  lddb(1, ldb_);
    std::vector<scalar_t> alpha(1, alpha_);

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, batch * size_A, A );  // TODO: generate
    lapack_larnv( idist, iseed, batch * size_B, B );  // TODO
    lapack_lacpy( "g", Bm, batch * Bn, B, ldb_, Bref, ldb_ );

    // set unused data to nan
    /*if (uplo_ == Uplo::Lower) {
        for (int64_t j = 0; j < Am; ++j)
            for (int64_t i = 0; i < j; ++i)  // upper
                A[ i + j*lda_ ] = nan("");
    }
    else {
        for (int64_t j = 0; j < Am; ++j)
            for (int64_t i = j+1; i < Am; ++i)  // lower
                A[ i + j*lda_ ] = nan("");
    }*/

    // Factor A into L L^H or U U^H to get a well-conditioned triangular matrix.
    // If diag_ == Unit, the diag_onal is replaced; this is still well-conditioned.
    for (size_t s = 0; s < batch; ++s) {
        TA* pA = Aarray[s];
        // First, brute force positive definiteness.
        for (int64_t i = 0; i < Am; ++i) {
            pA[ i + i*lda_ ] += Am;
        }
        int64_t potrf_info = 0;
        lapack_potrf( uplo2str(uplo_), Am, pA, lda_, &potrf_info );
        require( potrf_info == 0 );
    }

    // if row-major, transpose A
    if (layout == Layout::RowMajor) {
        for (size_t s = 0; s < batch; ++s) {
            for (int64_t j = 0; j < Am; ++j) {
                for (int64_t i = 0; i < j; ++i) {
                    swap( Aarray[s][ i + j*lda_ ], Aarray[s][ j + i*lda_ ] );
                }
            }
        }
    }

    blas::device_copy_matrix(Am, batch * Am, A, lda_, dA, lda_, queue);
    blas::device_copy_matrix(Bm, batch * Bn, B, ldb_, dB, ldb_, queue);
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t* Anorm = new real_t[ batch ];
    real_t* Bnorm = new real_t[ batch ];
    for (size_t s = 0; s < batch; ++s) {
        Anorm[ s ] = lapack_lantr( "f", uplo2str(uplo_), diag2str(diag_), Am, Am, Aarray[s], lda_, work );
        Bnorm[ s ] = lapack_lange( "f", Bm, Bn, Barray[s], ldb_, work );
    }

    // decide error checking mode
    info.resize( 0 );

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::batch::trsm( layout, side, uplo, trans, diag, m, n, alpha, dAarray, ldda, dBarray, lddb,
                       batch, info, queue );
    queue.sync();
    time = get_wtime() - time;

    double gflop = batch * blas::Gflop< scalar_t >::trsm( side_, m_, n_ );
    params.time()   = time;
    params.gflops() = gflop / time;

    blas::device_copy_matrix(Bm, batch * Bn, dB, ldb_, B, ldb_, queue);
    queue.sync();

    if (params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (size_t i = 0; i < batch; ++i) {
            cblas_trsm( cblas_layout_const(layout),
                        cblas_side_const(side_),
                        cblas_uplo_const(uplo_),
                        cblas_trans_const(trans_),
                        cblas_diag_const(diag_),
                        m_, n_, alpha_, Aarray[i], lda_, Brefarray[i], ldb_ );
        }
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        // check error compared to reference
        // Am is reduction dimension
        // beta = 0, Cnorm = 0 (initial).
        real_t err, error = 0;
        bool ok, okay = true;
        for (size_t i = 0; i < batch; ++i) {
            check_gemm( Bm, Bn, Am, alpha_, scalar_t(0), Anorm[i], Bnorm[i], real_t(0),
                        Brefarray[i], ldb_, Barray[i], ldb_, verbose, &err, &ok );
            error = std::max( error, err );
            okay &= ok;
        }
        params.error() = error;
        params.okay() = okay;
    }

    delete[] A;
    delete[] B;
    delete[] Bref;
    delete[] Anorm;
    delete[] Bnorm;

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
}

// -----------------------------------------------------------------------------
void test_batch_trsm_device( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_device_batch_trsm_work< float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_device_batch_trsm_work< double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_device_batch_trsm_work< std::complex<float>, std::complex<float> >
                ( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_device_batch_trsm_work< std::complex<double>, std::complex<double> >
                ( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
