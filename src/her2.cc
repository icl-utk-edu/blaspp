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
/// @ingroup her2_internal
inline void her2(
    char uplo,
    blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> const* y, blas_int incy,
    std::complex<float>*       A, blas_int lda )
{
    BLAS_cher2( &uplo, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx,
                (blas_complex_float*) y, &incy,
                (blas_complex_float*) A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup her2_internal
inline void her2(
    char uplo,
    blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> const* y, blas_int incy,
    std::complex<double>*       A, blas_int lda )
{
    BLAS_zher2( &uplo, &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx,
                (blas_complex_double*) y, &incy,
                (blas_complex_double*) A, &lda );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup her2_internal
///
template <typename scalar_t>
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    scalar_t alpha,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy,
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
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::her2_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::her2 );
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
    internal::her2( uplo_, n_,
                    alpha, x, incx_, y, incy_, A, lda_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda )
{
    blas::syr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda )
{
    blas::syr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda )
{
    impl::her2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda )
{
    impl::her2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

}  // namespace blas
