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
/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float beta,
    float       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < n );
    else
        blas_error_if( lda < k );

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_ssyrk( &uplo_, &trans_, &n_, &k_,
               &alpha, A, &lda_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double beta,
    double       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < n );
    else
        blas_error_if( lda < k );

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_dsyrk( &uplo_, &trans_, &n_, &k_,
               &alpha, A, &lda_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < n );
    else
        blas_error_if( lda < k );

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_csyrk( &uplo_, &trans_, &n_, &k_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < n );
    else
        blas_error_if( lda < k );

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_zsyrk( &uplo_, &trans_, &n_, &k_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &ldc_ );
}

}  // namespace blas
