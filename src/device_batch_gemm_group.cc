// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
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
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
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

    // gemm needs 3 arrays (A, B, and C).
    size_t max_chunk = MaxBatchChunk;
    queue.work_ensure_size<void*>( 3*max_chunk );

    scalar_t** dAarray = (scalar_t**) queue.work();
    scalar_t** dBarray = dAarray + max_chunk;
    scalar_t** dCarray = dBarray + max_chunk;

    // If we have only one group, no need to fork.
    bool do_fork = group_count > 1;
    if (do_fork)
        queue.fork();

    size_t grp_begin = 0;  // First group in this chunk.
    size_t ptr_begin = 0;  // First [ABC]array pointer in this chunk.
    size_t part_done = 0;  // gemms already done in current grp_begin.
    while (grp_begin < group_count) {
        // Find groups that fit into this chunk.
        size_t grp_end = grp_begin;
        size_t chunk_size = 0;
        while (grp_end < group_count) {
            if (grp_begin == grp_end) {
                // Part or whole of first group.
                chunk_size = min( group_size[ grp_end ] - part_done, max_chunk );
                grp_end += 1;
            }
            else if (chunk_size + group_size[ grp_end ] < max_chunk) {
                // Next group fits.
                chunk_size += group_size[ grp_end ];
                grp_end += 1;
            }
            else {
                // Next group doesn't fit.
                break;
            }
        }

        // Copy Aarray, Barray, and Carray to device.
        device_copy_vector(
            chunk_size, &Aarray[ ptr_begin ], 1, dAarray, 1, queue );
        device_copy_vector(
            chunk_size, &Barray[ ptr_begin ], 1, dBarray, 1, queue );
        device_copy_vector(
            chunk_size, &Carray[ ptr_begin ], 1, dCarray, 1, queue );

        // Launch kernels in this chunk.
        if (do_fork)
            queue.fork();
        size_t dev_ptr = 0;
        for (size_t ig = grp_begin; ig < grp_end; ++ig) {
            size_t ibatch_size;
            if (ig == grp_begin) {
                if (group_size[ ig ] - part_done > max_chunk) {
                    // Do first max_chunk part of first (and only) group.
                    ibatch_size = max_chunk;
                    part_done += ibatch_size;
                    assert( grp_end == grp_begin+1 );
                }
                else {
                    // Remainder of first group fits.
                    // Reset part_done since grp_begin will be updated.
                    ibatch_size = group_size[ ig ] - part_done;
                    part_done = 0;
                }
            }
            else {
                ibatch_size = group_size[ ig ];
            }

            // Extract params for the current group.
            blas::Op        transA_ = transA[ ig ];
            blas::Op        transB_ = transB[ ig ];
            device_blas_int m_      = to_device_blas_int( m[ ig ] );
            device_blas_int n_      = to_device_blas_int( n[ ig ] );
            device_blas_int k_      = to_device_blas_int( k[ ig ] );
            device_blas_int lda_    = to_device_blas_int( lda[ ig ] );
            device_blas_int ldb_    = to_device_blas_int( ldb[ ig ] );
            device_blas_int ldc_    = to_device_blas_int( ldc[ ig ] );

            if (layout == Layout::RowMajor) {
                // swap transA <=> transB, m <=> n, B <=> A
                internal::batch_gemm(
                    transB_, transA_, n_, m_, k_,
                    alpha[ig], &dBarray[ dev_ptr ], ldb_,
                               &dAarray[ dev_ptr ], lda_,
                    beta[ig],  &dCarray[ dev_ptr ], ldc_,
                    ibatch_size, queue );
            }
            else {
                internal::batch_gemm(
                    transA_, transB_, m_, n_, k_,
                    alpha[ig], &dAarray[ dev_ptr ], lda_,
                               &dBarray[ dev_ptr ], ldb_,
                    beta[ig],  &dCarray[ dev_ptr ], ldc_,
                    ibatch_size, queue );
            }
            dev_ptr += ibatch_size;

            if (do_fork)
                queue.revolve();
        }
        if (do_fork)
            queue.join();

        // If part_done, grp_begin isn't done, so don't update it;
        // otherwise, update grp_begin.
        if (part_done == 0)
            grp_begin = grp_end;
        ptr_begin += chunk_size;
    }
#endif
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
