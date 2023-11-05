// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "blas/flops.hh"
#include "print_matrix.hh"
#include "check_gemm.hh"


//------------------------------------------------------------------------------
// Copy A from LAPACK format on host to tile format on device.
// Each tile is dimension mb-by-nb in ld_tile-by-nb array, ld_tile >= mb.
// The matrix A is mt block rows by nt block cols,
// with overall dimension mb*mt-by-nb*nt in an lda-by-nb*nt array, lda >= mb*mt.
template <typename T>
void copy_lapack_to_tile_format(
    int64_t mb, int64_t nb, int64_t mt, int64_t nt,
    T const* A, int64_t lda,
    T* dA, int64_t ld_tile, blas::Queue& queue )
{
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            blas::device_copy_matrix(
                mb, nb,
                & A[ i*mb + j*nb*lda ], lda,
                &dA[ (i + j*mt)*nb*ld_tile ], ld_tile,
                queue );
        }
    }
}

//------------------------------------------------------------------------------
// Copy A from tile format on device to LAPACK format on host.
// See copy_lapack_to_tile_format for format.
template <typename T>
void copy_tile_to_lapack_format(
    int64_t mb, int64_t nb, int64_t mt, int64_t nt,
    T const* dA, int64_t ld_tile,
    T* A, int64_t lda, blas::Queue& queue )
{
    for (int64_t j = 0; j < nt; ++j) {
        for (int64_t i = 0; i < mt; ++i) {
            blas::device_copy_matrix(
                mb, nb,
                &dA[ (i + j*mt)*nb*ld_tile ], ld_tile,
                &A[ i*mb + j*nb*lda ], lda,
                queue );
        }
    }
}

//------------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
void test_schur_gemm_work( Params& params, bool run )
{
    using namespace testsweeper;
    using namespace blas::batch;
    using blas::Op;
    using blas::Layout;
    using blas::Format;
    using scalar_t = blas::scalar_type< TA, TB, TC >;
    using real_t   = blas::real_type< scalar_t >;

    // get & mark input values
    blas::Layout layout = Layout::ColMajor; //params.layout();
    blas::Format format = params.format();
    blas::Op transA_    = params.transA();
    blas::Op transB_    = params.transB();
    scalar_t alpha_     = params.alpha();
    scalar_t beta_      = params.beta();
    int64_t m_          = params.dim.m();
    int64_t n_          = params.dim.n();
    int64_t k_          = params.dim.k(); // Used as the tile size nb.
    int64_t device  = params.device();
    int64_t align   = params.align();
    int64_t verbose = params.verbose();

    // mark non-standard output values
    params.gflops();
    params.time2();
    params.gflops2();
    params.ref_time();
    params.ref_gflops();

    params.time   .name( "batch time (s)"  );
    params.gflops .name( "batch gflop/s"   );
    params.time2  .name( "stream time (s)" );
    params.gflops2.name( "stream gflop/s"  );

    if (! run)
        return;

    if (blas::get_device_count() == 0) {
        params.msg() = "skipping: no GPU devices or no GPU support";
        return;
    }

    // Round m_ and n_ down to a multiple of k, since we are not dealing with
    // cleanup of partial tiles around the edge of the matrix.
    m_ = int64_t( m_ / k_ ) * k_;
    n_ = int64_t( n_ / k_ ) * k_;
    params.dim.m() = m_;
    params.dim.n() = n_;

    // setup
    int64_t Am = (transA_ == Op::NoTrans ? m_ : k_);
    int64_t An = (transA_ == Op::NoTrans ? k_ : m_);
    int64_t Bm = (transB_ == Op::NoTrans ? k_ : n_);
    int64_t Bn = (transB_ == Op::NoTrans ? n_ : k_);
    int64_t Cm = m_;
    int64_t Cn = n_;
    // if (layout == Layout::RowMajor) {
    //     std::swap( Am, An );
    //     std::swap( Bm, Bn );
    //     std::swap( Cm, Cn );
    // }

    int64_t mt = int64_t( m_ / k_ );
    int64_t nt = int64_t( n_ / k_ );
    size_t batch = mt*nt;

    int64_t lda_ = roundup( Am, align );
    int64_t ldb_ = roundup( Bm, align );
    int64_t ldc_ = roundup( Cm, align );
    // ld of a tile. For now, there is
    // no padding for tiles.
    int64_t ld_tile = k_;
    size_t size_A = size_t(lda_)*An;
    size_t size_B = size_t(ldb_)*Bn;
    size_t size_C = size_t(ldc_)*Cn;
    TA* A    = new TA[ size_A ];
    TB* B    = new TB[ size_B ];
    TC* C    = new TC[ size_C ];
    TC* Cref = nullptr;
    if (params.ref() == 'y' || params.check() == 'y')
        Cref = new TC[ size_C ];

    // device specifics
    blas::Queue queue( device );
    TA* dA = blas::device_malloc<TA>( size_A, queue );
    TB* dB = blas::device_malloc<TB>( size_B, queue );
    TC* dC = blas::device_malloc<TC>( size_C, queue );

    // pointer arrays
    std::vector<TA*> dAarray;
    std::vector<TB*> dBarray;
    std::vector<TC*> dCarray;

    // wrap scalar arguments in std::vector
    std::vector<blas::Op> transA( 1, transA_ );
    std::vector<blas::Op> transB( 1, transB_ );
    std::vector<int64_t>  k( 1, k_ );
    int64_t lda_batch = lda_;
    int64_t ldb_batch = ldb_;
    int64_t ldc_batch = ldc_;
    if (format == Format::Tile) {
        lda_batch = ld_tile;
        ldb_batch = ld_tile;
        ldc_batch = ld_tile;
    }
    std::vector<int64_t>  ldda( 1, lda_batch );
    std::vector<int64_t>  lddb( 1, ldb_batch );
    std::vector<int64_t>  lddc( 1, ldc_batch );
    std::vector<scalar_t> alpha( 1, alpha_ );
    std::vector<scalar_t> beta( 1, beta_ );

    int64_t idist = 1;
    int iseed[4] = { 0, 0, 0, 1 };
    lapack_larnv( idist, iseed, size_A, A );
    lapack_larnv( idist, iseed, size_B, B );
    lapack_larnv( idist, iseed, size_C, C );
    if (Cref != nullptr)
        lapack_lacpy( "g", Cm, Cn, C, ldc_, Cref, ldc_ );

    if (format == Format::LAPACK) {
        blas::device_copy_matrix( Am, An, A, lda_, dA, lda_, queue );
        blas::device_copy_matrix( Bm, Bn, B, ldb_, dB, ldb_, queue );
        blas::device_copy_matrix( Cm, Cn, C, ldc_, dC, ldc_, queue );
    }
    else if (format == Format::Tile) {
        copy_lapack_to_tile_format(
                k_, k_, Am/k_, An/k_, A, lda_, dA, ld_tile, queue );
        copy_lapack_to_tile_format(
                k_, k_, Bm/k_, Bn/k_, B, ldb_, dB, ld_tile, queue );
        copy_lapack_to_tile_format(
                k_, k_, mt, nt, C, ldc_, dC, ld_tile, queue );
    }
    queue.sync();

    // norms for error check
    real_t work[1];
    real_t Anorm = lapack_lange( "f", Am, An, A, lda_, work );
    real_t Bnorm = lapack_lange( "f", Bm, Bn, B, ldb_, work );
    real_t Cnorm = lapack_lange( "f", Cm, Cn, C, ldc_, work );

    // Construct dAarray, dBarray, dCarray (on host) with pointers to
    // tiles in dA, dB, dC.
    double time_with_setup = get_wtime();
    if (format == Format::LAPACK) {
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                if (transA_ == Op::NoTrans)
                    dAarray.push_back( &dA[ i*k_ ] );          // i-th block row
                else
                    dAarray.push_back( &dA[ i*k_*lda_ ] );     // i-th block col

                if (transB_ == Op::NoTrans)
                    dBarray.push_back( &dB[ j*k_*ldb_ ] );     // j-th block col
                else
                    dBarray.push_back( &dB[ j*k_  ] );         // j-th block row

                dCarray.push_back( &dC[ i*k_ + j*k_*ldc_ ] );  // (i, j)-th block
            }
        }
    }
    else if (format == Format::Tile) {
        for (int64_t j = 0; j < nt; ++j) {
            for (int64_t i = 0; i < mt; ++i) {
                dAarray.push_back( &dA[ i*k_*ld_tile ] );  // i-th tile
                dBarray.push_back( &dB[ j*k_*ld_tile ] );  // j-th tile
                dCarray.push_back( &dC[ (i + j*mt)*k_*ld_tile ] );  // (i, j)-th tile
            }
        }
    }

    //----------------------------------------
    // Run batch test.
    // todo: warm up queue for batch.
    testsweeper::flush_cache( params.cache() );
    std::vector<int64_t> info;  // empty info vector (no checks)
    double time = get_wtime();
    blas::batch::gemm( layout, transA, transB, k, k, k, alpha, dAarray, ldda,
                       dBarray, lddb, beta, dCarray, lddc, batch, info, queue );
    queue.sync();
    double t = get_wtime();
    time_with_setup = t - time_with_setup;
    time = t - time;

    double gflop = blas::Gflop< scalar_t >::gemm( m_, n_, k_ );
    params.time()   = time;
    params.gflops() = gflop / time;

    if (format == Format::LAPACK) {
        blas::device_copy_matrix( Cm, Cn, dC, ldc_, C, ldc_, queue );
    }
    else if (format == Format::Tile) {
        copy_tile_to_lapack_format(
            k_, k_, mt, nt, dC, ld_tile, C, ldc_, queue );
    }
    queue.sync();

    //----------------------------------------
    // Run multi-stream test.
    // todo: warm up queue for streams.
    testsweeper::flush_cache( params.cache() );
    time = get_wtime();
    queue.fork();
    for (size_t i = 0; i < dCarray.size(); ++i) {
        blas::gemm( layout, transA_, transB_, k_, k_, k_,
                    alpha_, dAarray[ i ], lda_batch,
                            dBarray[ i ], ldb_batch,
                    beta_,  dCarray[ i ], ldc_batch, queue );
        queue.revolve();
    }
    queue.join();
    queue.sync();
    time = get_wtime() - time;

    params.time2()   = time;
    params.gflops2() = gflop / time;

    // todo: copy & check multi-stream result.

    if (params.ref() == 'y' || params.check() == 'y') {
        testsweeper::flush_cache( params.cache() );
        if (format == Format::Tile) {
            // Copy A and B to device in LAPACK format.
            blas::device_copy_matrix( Am, An, A, lda_, dA, lda_, queue );
            blas::device_copy_matrix( Bm, Bn, B, ldb_, dB, ldb_, queue );
        }
        blas::device_copy_matrix( Cm, Cn, Cref, ldc_, dC, ldc_, queue );
        queue.sync();

        //----------------------------------------
        // Run reference
        double time_ref = get_wtime();
        blas::gemm( layout, transA_, transB_, m_, n_, k_,
                    alpha_, dA, lda_, dB, ldb_,
                    beta_, dC, ldc_, queue );
        queue.sync();
        time_ref = get_wtime() - time_ref;
        params.ref_time()   = time_ref;
        params.ref_gflops() = gflop / time_ref;

        blas::device_copy_matrix( Cm, Cn, dC, ldc_, Cref, ldc_, queue );
        queue.sync();

        // Error
        real_t error;
        bool okay;
        check_gemm( Cm, Cn, k_, alpha_, beta_, Anorm, Bnorm, Cnorm,
                    Cref, ldc_, C, ldc_, verbose, &error, &okay );

        params.error() = error;
        params.okay() = okay;

        delete[] Cref;
    }

    delete[] A;
    delete[] B;
    delete[] C;

    blas::device_free( dA, queue );
    blas::device_free( dB, queue );
    blas::device_free( dC, queue );
}

// -----------------------------------------------------------------------------
void test_schur_gemm( Params& params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_schur_gemm_work< float, float, float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_schur_gemm_work< double, double, double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_schur_gemm_work< std::complex<float>, std::complex<float>,
                            std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_schur_gemm_work< std::complex<double>, std::complex<double>,
                            std::complex<double> >( params, run );
            break;

        default:
            throw std::exception();
            break;
    }
}
