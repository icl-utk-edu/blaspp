// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas_internal.hh"
#include "blas/counter.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup trmm_internal
inline void trmm(
    char side,
    char uplo,
    char trans,
    char diag,
    blas_int m, blas_int n,
    float alpha,
    float const* A, blas_int lda,
    float*       B, blas_int ldb )
{
    BLAS_strmm( &side, &uplo, &trans, &diag, &m, &n,
                &alpha, A, &lda, B, &ldb );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup trmm_internal
inline void trmm(
    char side,
    char uplo,
    char trans,
    char diag,
    blas_int m, blas_int n,
    double alpha,
    double const* A, blas_int lda,
    double*       B, blas_int ldb )
{
    BLAS_dtrmm( &side, &uplo, &trans, &diag, &m, &n,
                &alpha, A, &lda, B, &ldb );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup trmm_internal
inline void trmm(
    char side,
    char uplo,
    char trans,
    char diag,
    blas_int m, blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* A, blas_int lda,
    std::complex<float>*       B, blas_int ldb )
{
    BLAS_ctrmm( &side, &uplo, &trans, &diag, &m, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda,
                (blas_complex_float*) B, &ldb );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup trmm_internal
inline void trmm(
    char side,
    char uplo,
    char trans,
    char diag,
    blas_int m, blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* A, blas_int lda,
    std::complex<double>*       B, blas_int ldb )
{
    BLAS_ztrmm( &side, &uplo, &trans, &diag, &m, &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda,
                (blas_complex_double*) B, &ldb );
}

}  // namespace internal

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
    scalar_t*       B, int64_t ldb )
{
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
        counter::trmm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { side, uplo, trans, diag, m, n };
        counter::insert( element, counter::Id::trmm );
    #endif

    // convert arguments
    blas_int m_   = to_blas_int( m );
    blas_int n_   = to_blas_int( n );
    blas_int lda_ = to_blas_int( lda );
    blas_int ldb_ = to_blas_int( ldb );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        swap( m_, n_ );
    }
    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );

    // call low-level wrapper
    internal::trmm( side_, uplo_, trans_, diag_, m_, n_,
                    alpha, A, lda_, B, ldb_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
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
    float*       B, int64_t ldb )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb );
}

//------------------------------------------------------------------------------
/// CPU, double version.
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
    double*       B, int64_t ldb )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
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
    std::complex<float>*       B, int64_t ldb )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
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
    std::complex<double>*       B, int64_t ldb )
{
    impl::trmm( layout, side, uplo, trans, diag, m, n,
                alpha, A, lda, B, ldb );
}

}  // namespace blas
