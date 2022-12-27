// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/batch_common.hh"
#include "blas.hh"


namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched version.
/// Mid-level templated wrapper checks and converts arguments,
/// then makes individual routine calls in parallel.
/// @ingroup syr2k_internal
///
template <typename scalar_t>
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<scalar_t >  const& alpha,
    std::vector<scalar_t*>  const& Aarray, std::vector<int64_t> const& lda,
    std::vector<scalar_t*>  const& Barray, std::vector<int64_t> const& ldb,
    std::vector<scalar_t >  const& beta,
    std::vector<scalar_t*>  const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch_size < 0 );
    blas_error_if( info.size() != 0
                   && info.size() != 1
                   && info.size() != batch_size );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::syr2k_check(
            layout, uplo, trans, n, k,
            alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
            batch_size, info );
    }

    blas::internal_set_device( queue.device() );

    queue.fork();
    for (size_t i = 0; i < batch_size; ++i) {
        blas::Uplo uplo_   = blas::batch::extract( uplo,   i );
        blas::Op   trans_  = blas::batch::extract( trans,  i );
        int64_t    n_      = blas::batch::extract( n,      i );
        int64_t    k_      = blas::batch::extract( k,      i );
        int64_t    lda_    = blas::batch::extract( lda,    i );
        int64_t    ldb_    = blas::batch::extract( ldb,    i );
        int64_t    ldc_    = blas::batch::extract( ldc,    i );
        scalar_t   alpha_  = blas::batch::extract( alpha,  i );
        scalar_t   beta_   = blas::batch::extract( beta,   i );
        scalar_t*  A_      = blas::batch::extract( Aarray, i );
        scalar_t*  B_      = blas::batch::extract( Barray, i );
        scalar_t*  C_      = blas::batch::extract( Carray, i );
        blas::syr2k( layout, uplo_, trans_, n_, k_,
                     alpha_, A_, lda_, B_, ldb_, beta_, C_, ldc_,
                     queue );
        queue.revolve();
    }
    queue.join();
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
namespace batch {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, float version.
/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::syr2k( layout, uplo, trans, n, k,
                 alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                 batch_size, info, queue );
}

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::syr2k( layout, uplo, trans, n, k,
                 alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                 batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<float> version.
/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::syr2k( layout, uplo, trans, n, k,
                 alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                 batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<double> version.
/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::syr2k( layout, uplo, trans, n, k,
                 alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                 batch_size, info, queue );
}

}  // namespace batch
}  // namespace blas
