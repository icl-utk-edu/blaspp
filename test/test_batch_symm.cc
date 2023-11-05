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
template <typename TA, typename TB, typename TC>
void test_batch_symm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using blas::Side;
    using blas::Layout;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = params.layout();
    blas::Side side_     = params.side();
    blas::Uplo uplo_     = params.uplo();
    scalar_t alpha_      = params.alpha();
    scalar_t beta_       = params.beta();
    int64_t m_           = params.dim.m();
    int64_t n_           = params.dim.n();
    size_t  batch       = params.batch();
    int64_t align       = params.align();
    int64_t verbose     = params.verbose();

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();

    if (! run)
        return;

    // setup
    int64_t An = (side_ == Side::Left ? m_ : n_);
    int64_t Cm = m_;
    int64_t Cn = n_;
    if (layout == Layout::RowMajor)
        std::swap( Cm, Cn );
    int64_t lda_ = roundup( An, align );
    int64_t ldb_ = roundup( Cm, align );
    int64_t ldc_ = roundup( Cm, align );
    size_t size_A = size_t(lda_)*An;
    size_t size_B = size_t(ldb_)*Cn;
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

    for (size_t s = 0; s < batch; ++s) {
         Aarray[s]   =  A   + s * size_A;
         Barray[s]   =  B   + s * size_B;
         Carray[s]   =  C   + s * size_C;
        Crefarray[s] = Cref + s * size_C;
    }

    // info
    std::vector<int64_t> info( batch );

    // wrap scalar arguments in std::vector
    std::vector<blas::Side> side(1, side_);
    std::vector<blas::Uplo> uplo(1, uplo_);
    std::vector<int64_t>    m(1, m_);
    std::vector<int64_t>    n(1, n_);
    std::vector<int64_t>    lda(1, lda_);
    std::vector<int64_t>    ldb(1, ldb_);
    std::vector<int64_t>    ldc(1, ldc_);
    std::vector<scalar_t>   alpha(1, alpha_);
    std::vector<scalar_t>   beta(1, beta_);

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

    for (size_t s = 0; s < batch; ++s) {
        Anorm[s] = lapack_lansy( "f", uplo2str(uplo_), An, Aarray[s], lda_, work );
        Bnorm[s] = lapack_lange( "f", Cm, Cn, Barray[s], ldb_, work );
        Cnorm[s] = lapack_lange( "f", Cm, Cn, Carray[s], ldc_, work );
    }

    // decide error checking mode
    info.resize( 0 );

    // run test
    testsweeper::flush_cache( params.cache() );
    double time = get_wtime();
    blas::batch::symm( layout, side, uplo, m, n, alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                       batch, info );
    time = get_wtime() - time;

    double gflop = batch * blas::Gflop< scalar_t >::symm( side_, m_, n_ );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (params.ref() == 'y' || params.check() == 'y') {
        // run reference
        testsweeper::flush_cache( params.cache() );
        time = get_wtime();
        for (size_t s = 0; s < batch; ++s) {
            cblas_symm( cblas_layout_const(layout),
                        cblas_side_const(side_),
                        cblas_uplo_const(uplo_),
                        m_, n_, alpha_, Aarray[s], lda_, Barray[s], ldb_, beta_, Crefarray[s], ldc_ );
        }
        time = get_wtime() - time;

        params.ref_time()   = time;
        params.ref_gflops() = gflop / time;

        // check error compared to reference
        real_t err, error = 0;
        bool ok, okay = true;
        for (size_t s = 0; s < batch; ++s) {
            check_gemm( Cm, Cn, An, alpha_, beta_, Anorm[s], Bnorm[s], Cnorm[s],
                        Crefarray[s], ldc_, Carray[s], ldc_, verbose, &err, &ok );
            error = std::max( error, err );
            okay &= ok;
        }
        params.error() = error;
        params.okay() = okay;
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
void test_batch_symm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_batch_symm_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_batch_symm_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_batch_symm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_batch_symm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
