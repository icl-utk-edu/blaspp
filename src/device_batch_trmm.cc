// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/batch_common.hh"
#include "blas.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched version.
/// Mid-level templated wrapper checks and converts arguments,
/// then makes individual routine calls in parallel.
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(
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
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch_size < 0 );
    blas_error_if( info.size() != 0
                   && info.size() != 1
                   && info.size() != batch_size );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trmm_check(
            layout, side, uplo, trans, diag, m, n,
            alpha, Aarray, lda, Barray, ldb, batch_size, info );
    }

    // rocm 4.0 seems to have bug with trmm when using multiple streams.
    #ifdef BLAS_HAVE_ROCBLAS
        const bool fork = false;
    #else
        const bool fork = true;
    #endif

    blas::internal_set_device( queue.device() );

    if (fork)
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
        blas::trmm( layout, side_, uplo_, trans_, diag_, m_, n_,
                    alpha_, A_, lda_, B_, ldb_, queue );
        if (fork)
            queue.revolve();
    }
    if (fork)
        queue.join();
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
namespace batch {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, float version.
/// @ingroup trmm
void trmm(
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
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, double version.
/// @ingroup trmm
void trmm(
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
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<float> version.
/// @ingroup trmm
void trmm(
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
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<double> version.
/// @ingroup trmm
void trmm(
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
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, Aarray, lda, Barray, ldb,
                batch_size, info, queue );
}

}  // namespace batch
}  // namespace blas
