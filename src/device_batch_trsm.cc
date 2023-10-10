// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/batch_common.hh"
#include "blas.hh"

#include "device_internal.hh"

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched version.
/// Mid-level templated wrapper checks and converts arguments,
/// then makes individual routine calls in parallel.
/// @ingroup trsm_internal
///
template <typename scalar_t>
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<scalar_t >  const& alpha,
    std::vector<scalar_t*>  const& Aarray, std::vector<int64_t> const& lda,
    std::vector<scalar_t*>  const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    using std::swap;

    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch_size < 0 );
    blas_error_if( info.size() != 0
                   && info.size() != 1
                   && info.size() != batch_size );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trsm_check(
            layout, side, uplo, trans, diag, m, n,
            alpha, Aarray, lda, Barray, ldb,
            batch_size, info );
    }

    bool fixed_size = (
           side.size()   == 1
        && uplo.size()   == 1
        && trans.size()  == 1
        && diag.size()   == 1
        && m.size()      == 1
        && n.size()      == 1
        && alpha.size()  == 1
        && Aarray.size() == batch_size
        && lda.size()    == 1
        && Barray.size() == batch_size
        && ldb.size()    == 1);

    blas::internal_set_device( queue.device() );

    if (fixed_size) {
        // convert arguments
        blas::Side side_  = side[0];
        blas::Uplo uplo_  = uplo[0];
        blas::Op   trans_ = trans[0];
        blas::Diag diag_  = diag[0];
        device_blas_int m_   = to_device_blas_int( m[0] );
        device_blas_int n_   = to_device_blas_int( n[0] );
        device_blas_int lda_ = to_device_blas_int( lda[0] );
        device_blas_int ldb_ = to_device_blas_int( ldb[0] );

        if (layout == Layout::RowMajor) {
            // swap lower <=> upper, left <=> right, m <=> n
            uplo_ = ( uplo_ == blas::Uplo::Lower ? blas::Uplo::Upper : blas::Uplo::Lower );
            side_ = ( side_ == blas::Side::Left ? blas::Side::Right : blas::Side::Left );
            swap( m_, n_ );
        }

        // trsm needs only 2 ptr arrays (A and B). Allocate usual
        // 3*max_chunk used by gemm, but then split in 2 instead of 3.
        size_t max_chunk = MaxBatchChunk;
        queue.work_ensure_size<void*>( 3*max_chunk );
        max_chunk = 3*max_chunk / 2;

        scalar_t** dAarray = (scalar_t**) queue.work();
        scalar_t** dBarray = dAarray + max_chunk;

        for (size_t i = 0; i < batch_size; i += max_chunk) {
            size_t ibatch_size = std::min( max_chunk, batch_size - i );

            // copy pointer array(s) to device
            device_copy_vector( ibatch_size, &Aarray[ i ], 1, dAarray, 1, queue );
            device_copy_vector( ibatch_size, &Barray[ i ], 1, dBarray, 1, queue );

            // call the vendor routine
            internal::batch_trsm(
                side_, uplo_, trans_, diag_, m_, n_,
                alpha[0], dAarray, lda_, dBarray, ldb_, ibatch_size,
                queue );
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch_size; ++i) {
            blas::Side side_   = blas::batch::extract( side,   i );
            blas::Uplo uplo_   = blas::batch::extract( uplo,   i );
            blas::Op   trans_  = blas::batch::extract( trans,  i );
            blas::Diag diag_   = blas::batch::extract( diag,   i );
            int64_t    m_      = blas::batch::extract( m,      i );
            int64_t    n_      = blas::batch::extract( n,      i );
            int64_t    lda_    = blas::batch::extract( lda,    i );
            int64_t    ldb_    = blas::batch::extract( ldb,    i );
            scalar_t   alpha_  = blas::batch::extract( alpha,  i );
            scalar_t*  A_      = blas::batch::extract( Aarray, i );
            scalar_t*  B_      = blas::batch::extract( Barray, i );
            blas::trsm( layout, side_, uplo_, trans_, diag_, m_, n_,
                        alpha_, A_, lda_, B_, ldb_,
                        queue );
            queue.revolve();
        }
        queue.join();
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
namespace batch {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, float version.
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::trsm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, double version.
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::trsm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<float> version.
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::trsm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<double> version.
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::trsm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

}  // namespace batch
}  // namespace blas
