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
/// @ingroup syr_internal
inline void syr(
    char uplo,
    blas_int n,
    float alpha,
    float const* x, blas_int incx,
    float*       A, blas_int lda )
{
    BLAS_ssyr( &uplo, &n, &alpha, x, &incx, A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup syr_internal
inline void syr(
    char uplo,
    blas_int n,
    double alpha,
    double const* x, blas_int incx,
    double*       A, blas_int lda )
{
    BLAS_dsyr( &uplo, &n, &alpha, x, &incx, A, &lda );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup syr_internal
///
template <typename scalar_t>
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    scalar_t alpha,
    scalar_t const* x, int64_t incx,
    scalar_t*       A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::syr_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::syr );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }
    char uplo_ = uplo2char( uplo );

    // call low-level wrapper
    internal::syr( uplo_, n_,
                   alpha, x, incx_, A, lda_ );
}

}  // namespace impl

//==============================================================================
// [cz]syr are in LAPACK, so those wrappers are in LAPACK++.
// todo: move [cz]syr back to BLAS++. We link with LAPACK now.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float*       A, int64_t lda )
{
    impl::syr( layout, uplo, n,
               alpha, x, incx, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double*       A, int64_t lda )
{
    impl::syr( layout, uplo, n,
               alpha, x, incx, A, lda );
}

}  // namespace blas
