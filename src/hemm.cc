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
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup hemm_internal
inline void hemm(
    char side,
    char uplo,
    blas_int m, blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* A, blas_int lda,
    std::complex<float> const* B, blas_int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, blas_int ldc )
{
    BLAS_chemm( &side, &uplo, &m, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda,
                (blas_complex_float*) B, &ldb,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup hemm_internal
inline void hemm(
    char side,
    char uplo,
    blas_int m, blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* A, blas_int lda,
    std::complex<double> const* B, blas_int ldb,
    std::complex<double> beta,
    std::complex<double>*       C, blas_int ldc )
{
    BLAS_zhemm( &side, &uplo, &m, &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda,
                (blas_complex_double*) B, &ldb,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &ldc );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup hemm_internal
///
template <typename scalar_t>
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* B, int64_t ldb,
    scalar_t beta,
    scalar_t*       C, int64_t ldc )
{
    using std::swap;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( lda < m, "lda %lld < m %lld",
                           llong( lda ), llong( m ) );
    else
        blas_error_if_msg( lda < n, "lda %lld < n %lld",
                           llong( lda ), llong( n ) );

    if (layout == Layout::ColMajor) {
        blas_error_if( ldb < m );
        blas_error_if( ldc < m );
    }
    else {
        blas_error_if( ldb < n );
        blas_error_if( ldc < n );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::hemm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { side, uplo, m, n };
        counter::insert( element, counter::Id::hemm );
    #endif

    // convert arguments
    blas_int m_   = to_blas_int( m );
    blas_int n_   = to_blas_int( n );
    blas_int lda_ = to_blas_int( lda );
    blas_int ldb_ = to_blas_int( ldb );
    blas_int ldc_ = to_blas_int( ldc );

    if (layout == Layout::RowMajor) {
        // swap left <=> right, lower <=> upper, m <=> n
        side = (side == Side::Left  ? Side::Right : Side::Left);
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        swap( m_, n_ );
    }
    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );

    // call low-level wrapper
    internal::hemm( side_, uplo_, m_, n_,
                    alpha, A, lda_, B, ldb_, beta, C, ldc_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// Same as blas::symm for float.
/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc )
{
    blas::symm( layout, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// Same as blas::symm for double.
/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc )
{
    blas::symm( layout, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc )
{
    impl::hemm( layout, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc )
{
    impl::hemm( layout, side, uplo, m, n,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

}  // namespace blas
