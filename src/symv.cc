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
/// @ingroup symv_internal
inline void symv(
    char uplo,
    blas_int n,
    float alpha,
    float const* A, blas_int lda,
    float const* x, blas_int incx,
    float beta,
    float*       y, blas_int incy )
{
    BLAS_ssymv( &uplo, &n,
                &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup symv_internal
inline void symv(
    char uplo,
    blas_int n,
    double alpha,
    double const* A, blas_int lda,
    double const* x, blas_int incx,
    double beta,
    double*       y, blas_int incy )
{
    BLAS_dsymv( &uplo, &n,
                &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup symv_internal
///
template <typename scalar_t>
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* x, int64_t incx,
    scalar_t beta,
    scalar_t*       y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::symv_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::symv );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }
    char uplo_ = uplo2char( uplo );

    // call low-level wrapper
    internal::symv( uplo_, n_,
                    alpha, A, lda_, x, incx_, beta, y, incy_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
// [cz]symv are in LAPACK, so those wrappers are in LAPACK++.
// todo: move [cz]symv back to BLAS++. We link with LAPACK now.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float*       y, int64_t incy )
{
    impl::symv( layout, uplo, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double*       y, int64_t incy )
{
    impl::symv( layout, uplo, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

}  // namespace blas
