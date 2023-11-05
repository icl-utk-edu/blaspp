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
/// CPU, variable-size batched version.
/// Mid-level templated wrapper checks and converts arguments,
/// then makes individual routine calls in parallel.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< real_type<scalar_t> > const& alpha,
    std::vector<scalar_t*>  const& Aarray, std::vector<int64_t> const& lda,
    std::vector< real_type<scalar_t> > const& beta,
    std::vector<scalar_t*>  const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info )
{
    using real_t = real_type<scalar_t>;

    blas_error_if( batch_size < 0 );
    blas_error_if( info.size() != 0
                   && info.size() != 1
                   && info.size() != batch_size );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::herk_check(
            layout, uplo, trans, n, k,
            alpha, Aarray, lda, beta, Carray, ldc,
            batch_size, info );
    }

    #pragma omp parallel for schedule( dynamic )
    for (size_t i = 0; i < batch_size; ++i) {
        blas::Uplo uplo_   = blas::batch::extract( uplo,   i );
        blas::Op   trans_  = blas::batch::extract( trans,  i );
        int64_t    n_      = blas::batch::extract( n,      i );
        int64_t    k_      = blas::batch::extract( k,      i );
        int64_t    lda_    = blas::batch::extract( lda,    i );
        int64_t    ldc_    = blas::batch::extract( ldc,    i );
        real_t     alpha_  = blas::batch::extract( alpha,  i );
        real_t     beta_   = blas::batch::extract( beta,   i );
        scalar_t*  A_      = blas::batch::extract( Aarray, i );
        scalar_t*  C_      = blas::batch::extract( Carray, i );
        blas::herk( layout, uplo_, trans_, n_, k_,
                    alpha_, A_, lda_, beta_, C_, ldc_ );
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
namespace batch {

//------------------------------------------------------------------------------
/// CPU, variable-size batched, float version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, Aarray, lda, beta, Carray, ldc,
                batch_size, info );
}

//------------------------------------------------------------------------------
/// CPU, variable-size batched, double version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, Aarray, lda, beta, Carray, ldc,
                batch_size, info );
}

//------------------------------------------------------------------------------
/// CPU, variable-size batched, complex<float> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< float >                const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< float >                const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, Aarray, lda, beta, Carray, ldc,
                batch_size, info );
}

//------------------------------------------------------------------------------
/// CPU, variable-size batched, complex<double> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< double >                const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< double >                const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, Aarray, lda, beta, Carray, ldc,
                batch_size, info );
}

}  // namespace batch
}  // namespace blas
