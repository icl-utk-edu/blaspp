// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* B, int64_t ldb,
    scalar_t beta,
    scalar_t*       C, int64_t ldc,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB == Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB != Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < n );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_gemm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { transA, transB, m, n, k };
        counter::insert( element, counter::Id::dev_gemm );
    #endif

    // convert arguments
    device_blas_int m_   = to_device_blas_int( m );
    device_blas_int n_   = to_device_blas_int( n );
    device_blas_int k_   = to_device_blas_int( k );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int ldb_ = to_device_blas_int( ldb );
    device_blas_int ldc_ = to_device_blas_int( ldc );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        internal::gemm( transB, transA, n_, m_, k_,
                        alpha, B, ldb_, A, lda_, beta, C, ldc_, queue );
    }
    else {
        internal::gemm( transA, transB, m_, n_, k_,
                        alpha, A, lda_, B, ldb_, beta, C, ldc_, queue );
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc, queue );
}

}  // namespace blas
