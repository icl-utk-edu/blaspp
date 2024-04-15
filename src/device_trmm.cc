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
/// @ingroup trmm_internal
///
template <typename scalar_t>
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t*       B, int64_t ldb,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    using std::swap;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldb < m );
    else
        blas_error_if( ldb < n );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_trmm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { side, uplo, trans, diag, m, n };
        counter::insert( element, counter::Id::dev_trmm );
    #endif

    // convert arguments
    device_blas_int m_   = to_device_blas_int( m );
    device_blas_int n_   = to_device_blas_int( n );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int ldb_ = to_device_blas_int( ldb );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        swap( m_, n_ );
    }

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::trmm( side, uplo, trans, diag, m_, n_,
                    alpha, A, lda_, B, ldb_, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float*       B, int64_t ldb,
    blas::Queue& queue )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double*       B, int64_t ldb,
    blas::Queue& queue )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       B, int64_t ldb,
    blas::Queue& queue )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       B, int64_t ldb,
    blas::Queue& queue )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb, queue );
}

}  // namespace blas
