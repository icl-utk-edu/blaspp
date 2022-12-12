// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/batch_common.hh"
#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// GPU device, group batched version.
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
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    size_t batch_size = 0;
    size_t group_count = group_size.size();
    if (group_count == 0)
        return;

    blas_error_if( layout != Layout::ColMajor
                   && layout != Layout::RowMajor );
    blas_error_if( info.size() != 0
                   && info.size() != group_count );

    for (size_t ig = 0; ig < group_count; ++ig) {
        batch_size += group_size[ ig ];
    }

    blas_error_if( transA.size() !=  group_count );
    blas_error_if( transB.size() !=  group_count );
    blas_error_if( m.size()      !=  group_count );
    blas_error_if( n.size()      !=  group_count );
    blas_error_if( k.size()      !=  group_count );
    blas_error_if( alpha.size()  !=  group_count );
    blas_error_if( lda.size()    !=  group_count );
    blas_error_if( ldb.size()    !=  group_count );
    blas_error_if( beta.size()   !=  group_count );
    blas_error_if( ldc.size()    !=  group_count );

    blas_error_if( Aarray.size() !=  batch_size );
    blas_error_if( Barray.size() !=  batch_size );
    blas_error_if( Carray.size() !=  batch_size );

    // assume at least one operation per group
    blas_error_if( batch_size < group_count );

    if (info.size() > 0) {
        // perform error checking
        blas::batch::gemm_check(
            layout, transA, transB, m, n, k,
            alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
            group_count, info );
    }

    blas::internal_set_device( queue.device() );

    scalar_t **dAarray, **dBarray, **dCarray;
    size_t batch_limit = queue.get_batch_limit();
    size_t processed = 0;

    // If we have only one group, no need to fork.
    if (group_count > 1)
        queue.fork();

    for (size_t ig = 0; ig < group_count; ig++) {
        // extract params for the current group
        size_t          batch   = group_size[ ig ];
        blas::Op        transA_ = transA[ ig ];
        blas::Op        transB_ = transB[ ig ];
        device_blas_int m_      = to_device_blas_int( m[ ig ] );
        device_blas_int n_      = to_device_blas_int( n[ ig ] );
        device_blas_int k_      = to_device_blas_int( k[ ig ] );
        device_blas_int lda_    = to_device_blas_int( lda[ ig ] );
        device_blas_int ldb_    = to_device_blas_int( ldb[ ig ] );
        device_blas_int ldc_    = to_device_blas_int( ldc[ ig ] );

        // Each group is submitted to a different stream using strides
        // of batch_limit.
        // First, get the device pointer array for the current stream.
        dAarray = ( scalar_t**) queue.get_dev_ptr_array( );
        dBarray = dAarray + batch_limit;
        dCarray = dBarray + batch_limit;

        for (size_t i = 0; i < batch_size; i += batch_limit) {
            size_t ibatch_size = std::min( batch_limit, batch_size - i );

            // copy Aarray, Barray, and Carray to device
            device_copy_vector(
                ibatch_size,& Aarray[ processed+i ], 1,
                dAarray, 1, queue );
            device_copy_vector(
                ibatch_size,& Barray[ processed+i ], 1,
                dBarray, 1, queue );
            device_copy_vector(
                ibatch_size,& Carray[ processed+i ], 1,
                dCarray, 1, queue );

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                internal::batch_gemm(
                    transB_, transA_, n_, m_, k_,
                    alpha[ig], dBarray, ldb_, dAarray, lda_,
                    beta[ig],  dCarray, ldc_,
                    ibatch_size, queue );
            }
            else {
                internal::batch_gemm(
                    transA_, transB_, m_, n_, k_,
                    alpha[ig], dAarray, lda_, dBarray, ldb_,
                    beta[ig],  dCarray, ldc_,
                    ibatch_size, queue );
            }
        }

        processed += batch;
        if (group_count > 1)
            queue.revolve();
    }

    if (group_count > 1)
        queue.join();
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
namespace batch {

//------------------------------------------------------------------------------
/// GPU device, group batched, float version.
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
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                group_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, group batched, double version.
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
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                group_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, group batched, complex<float> version.
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
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                group_size, info, queue );
}

//------------------------------------------------------------------------------
/// GPU device, group batched, complex<double> version.
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
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, Aarray, lda, Barray, ldb, beta, Carray, ldc,
                group_size, info, queue );
}

}  // namespace batch
}  // namespace blas
