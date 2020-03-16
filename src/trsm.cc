// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float       *B, int64_t ldb )
{
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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_strsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double       *B, int64_t ldb )
{
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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_dtrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *B, int64_t ldb )
{
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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_ctrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda_,
                (blas_complex_float*) B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *B, int64_t ldb )
{
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

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_ztrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda_,
                (blas_complex_double*) B, &ldb_ );
}

}  // namespace blas
