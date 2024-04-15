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
/// @ingroup her_internal
inline void her(
    char uplo,
    blas_int n,
    float alpha,
    std::complex<float> const* x, blas_int incx,
    std::complex<float>*       A, blas_int lda )
{
    BLAS_cher( &uplo, &n,
               &alpha,
               (blas_complex_float*) x, &incx,
               (blas_complex_float*) A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup her_internal
inline void her(
    char uplo,
    blas_int n,
    double alpha,
    std::complex<double> const* x, blas_int incx,
    std::complex<double>*       A, blas_int lda )
{
    BLAS_zher( &uplo, &n,
               &alpha,
               (blas_complex_double*) x, &incx,
               (blas_complex_double*) A, &lda );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup her_internal
///
template <typename scalar_t>
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
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
        counter::her_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::her );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );

    // Deal with layout. RowMajor needs copy of x in x2;
    // in other cases, x2 == x.
    scalar_t* x2 = const_cast< scalar_t* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate x (in x2)
        x2 = new scalar_t[ n ];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x2[ i ] = conj( x[ ix ] );
            ix += incx;
        }
        incx_ = 1;
    }
    char uplo_ = uplo2char( uplo );

    // call low-level wrapper
    internal::her( uplo_, n_,
                   alpha, x2, incx_, A, lda_ );

    if (layout == Layout::RowMajor) {
        delete[] x2;
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float*       A, int64_t lda )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double*       A, int64_t lda )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>*       A, int64_t lda )
{
    impl::her( layout, uplo, n,
               alpha, x, incx, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>*       A, int64_t lda )
{
    impl::her( layout, uplo, n,
               alpha, x, incx, A, lda );
}

}  // namespace blas
