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
/// @ingroup herk_internal
inline void herk(
    char uplo,
    char trans,
    blas_int n, blas_int k,
    float alpha,  // note: real
    std::complex<float> const* A, blas_int lda,
    float beta,   // note: real
    std::complex<float>*       C, blas_int ldc )
{
    BLAS_cherk( &uplo, &trans, &n, &k,
                &alpha,
                (blas_complex_float*) A, &lda,
                &beta,
                (blas_complex_float*) C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup herk_internal
inline void herk(
    char uplo,
    char trans,
    blas_int n, blas_int k,
    double alpha,  // note: real
    std::complex<double> const* A, blas_int lda,
    double beta,   // note: real
    std::complex<double>*       C, blas_int ldc )
{
    BLAS_zherk( &uplo, &trans, &n, &k,
                &alpha,
                (blas_complex_double*) A, &lda,
                &beta,
                (blas_complex_double*) C, &ldc );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup herk_internal
///
template <typename scalar_t>
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    real_type<scalar_t> alpha,  // note: real
    scalar_t const* A, int64_t lda,
    real_type<scalar_t> beta,   // note: real
    scalar_t*       C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < n );
    else
        blas_error_if( lda < k );

    blas_error_if( ldc < n );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::herk_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, trans, n, k };
        counter::insert( element, counter::Id::herk );
    #endif

    // convert arguments
    blas_int n_   = to_blas_int( n );
    blas_int k_   = to_blas_int( k );
    blas_int lda_ = to_blas_int( lda );
    blas_int ldc_ = to_blas_int( ldc );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^H; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);
    }
    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );

    // call low-level wrapper
    internal::herk( uplo_, trans_, n_, k_,
                    alpha, A, lda_, beta, C, ldc_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float beta,
    float*       C, int64_t ldc )
{
    blas::syrk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double beta,
    double*       C, int64_t ldc )
{
    blas::syrk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,  // note: real
    std::complex<float> const* A, int64_t lda,
    float beta,   // note: real
    std::complex<float>*       C, int64_t ldc )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    std::complex<double> const* A, int64_t lda,
    double beta,
    std::complex<double>*       C, int64_t ldc )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc );
}

}  // namespace blas
