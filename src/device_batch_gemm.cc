// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/batch_common.hh"
#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// GPU device, variable-size batched version.
/// Mid-level templated wrapper checks and converts arguments,
/// then makes individual routine calls in parallel.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
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
        blas::batch::gemm_check(
            layout, transA, transB, m, n, k,
            alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
            batch_size, info );
    }

    bool fixed_size = (
           transA.size() == 1
        && transB.size() == 1
        && m.size()      == 1
        && n.size()      == 1
        && k.size()      == 1
        && alpha.size()  == 1
        && Aarray.size() == batch_size
        && lda.size()    == 1
        && Barray.size() == batch_size
        && ldb.size()    == 1
        && beta.size()   == 1
        && Carray.size() == batch_size
        && ldc.size( )   == 1);

    blas::internal_set_device( queue.device() );

    if (fixed_size) {
        // convert arguments
        blas::Op transA_ = transA[0];
        blas::Op transB_ = transB[0];
        device_blas_int m_   = to_device_blas_int( m[0] );
        device_blas_int n_   = to_device_blas_int( n[0] );
        device_blas_int k_   = to_device_blas_int( k[0] );
        device_blas_int lda_ = to_device_blas_int( lda[0] );
        device_blas_int ldb_ = to_device_blas_int( ldb[0] );
        device_blas_int ldc_ = to_device_blas_int( ldc[0] );

        #ifdef BLAS_HAVE_PAPI
            // PAPI instrumentation
            counter::dev_batch_gemm_type element;
            memset( &element, 0, sizeof( element ) );
            element = { transA_, transB_, m_, n_, k_, batch_size };
            counter::insert( element, counter::Id::dev_batch_gemm );
        #endif

        // gemm needs 3 arrays (A, B, and C).
        size_t max_chunk = MaxBatchChunk;
        queue.work_ensure_size<void*>( 3*max_chunk );

        scalar_t** dAarray = (scalar_t**) queue.work();
        scalar_t** dBarray = dAarray + max_chunk;
        scalar_t** dCarray = dBarray + max_chunk;

        for (size_t i = 0; i < batch_size; i += max_chunk) {
            size_t ibatch_size = std::min( max_chunk, batch_size - i );

            // copy Aarray, Barray, and Carray to device
            device_copy_vector( ibatch_size, &Aarray[ i ], 1, dAarray, 1, queue );
            device_copy_vector( ibatch_size, &Barray[ i ], 1, dBarray, 1, queue );
            device_copy_vector( ibatch_size, &Carray[ i ], 1, dCarray, 1, queue );

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                internal::batch_gemm(
                    transB_, transA_, n_, m_, k_,
                    alpha[0], dBarray, ldb_, dAarray, lda_,
                    beta[0],  dCarray, ldc_,
                    ibatch_size, queue );
            }
            else {
                internal::batch_gemm(
                    transA_, transB_, m_, n_, k_,
                    alpha[0], dAarray, lda_, dBarray, ldb_,
                    beta[0],  dCarray, ldc_,
                    ibatch_size, queue );
            }
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch_size; ++i) {
            blas::Op   transA_ = blas::batch::extract( transA, i );
            blas::Op   transB_ = blas::batch::extract( transB, i );
            int64_t    m_      = blas::batch::extract( m,      i );
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
            blas::gemm( layout, transA_, transB_, m_, n_, k_,
                        alpha_, A_, lda_, B_, ldb_, beta_, C_, ldc_,
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
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
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
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, double version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
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
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<float> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
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
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                batch_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, variable-size batched, complex<double> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
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
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                batch_size, info, queue );
}

}  // namespace batch
}  // namespace blas
