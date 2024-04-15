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
/// @ingroup ger_internal
inline void ger(
    blas_int m, blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> const* y, blas_int incy,
    std::complex<float>*       A, blas_int lda )
{
    BLAS_cgerc( &m, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx,
                (blas_complex_float*) y, &incy,
                (blas_complex_float*) A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup ger_internal
inline void ger(
    blas_int m, blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> const* y, blas_int incy,
    std::complex<double>*       A, blas_int lda )
{
    BLAS_zgerc( &m, &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx,
                (blas_complex_double*) y, &incy,
                (blas_complex_double*) A, &lda );
}

//==============================================================================
// Unconjugated x y^T versions.

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float, unconjugated x y^T
/// version.
/// @ingroup geru_internal
inline void geru(
    blas_int m, blas_int n,
    float alpha,
    float const* x, blas_int incx,
    float const* y, blas_int incy,
    float*       A, blas_int lda )
{
    BLAS_sger( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double, unconjugated  x y^T
/// version.
/// @ingroup geru_internal
inline void geru(
    blas_int m, blas_int n,
    double alpha,
    double const* x, blas_int incx,
    double const* y, blas_int incy,
    double*       A, blas_int lda )
{
    BLAS_dger( &m, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float>, unconjugated x y^T
/// version.
/// @ingroup geru_internal
inline void geru(
    blas_int m, blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> const* y, blas_int incy,
    std::complex<float>*       A, blas_int lda )
{
    BLAS_cgeru( &m, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx,
                (blas_complex_float*) y, &incy,
                (blas_complex_float*) A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double>, unconjugated x y^T
/// version.
/// @ingroup geru_internal
inline void geru(
    blas_int m, blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> const* y, blas_int incy,
    std::complex<double>*       A, blas_int lda )
{
    BLAS_zgeru( &m, &n,
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
/// Conjugated x y^H version.
/// @ingroup ger_internal
///
template <typename scalar_t>
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy,
    scalar_t*       A, int64_t lda )
{
    static_assert( is_complex<scalar_t>::value, "complex version" );

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::ger_type element;
        memset( &element, 0, sizeof( element ) );
        element = { m, n };
        counter::insert( element, counter::Id::ger );
    #endif

    // convert arguments
    blas_int m_    = to_blas_int( m );
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    // call low-level wrapper
    if (layout == Layout::RowMajor) {
        // conjugate y (in y2)
        scalar_t* y2 = new scalar_t[ n ];
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y2[ i ] = conj( y[ iy ] );
            iy += incy;
        }
        incy_ = 1;

        // swap m <=> n, x <=> y, call geru
        internal::geru( n_, m_, alpha, y2, incy_, x, incx_, A, lda_ );

        delete[] y2;
    }
    else {
        internal::ger( m_, n_, alpha, x, incx_, y, incy_, A, lda_ );
    }
}

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// Unconjugated x y^T version.
/// @ingroup geru_internal
///
template <typename scalar_t>
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    scalar_t alpha,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy,
    scalar_t*       A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::geru_type element;
        memset( &element, 0, sizeof( element ) );
        element = { m, n };
        counter::insert( element, counter::Id::geru );
    #endif

    blas_int m_    = to_blas_int( m );
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    if (layout == Layout::RowMajor) {
        // swap m <=> n, x <=> y
        internal::geru( n_, m_, alpha, y, incy_, x, incx_, A, lda_ );
    }
    else {
        internal::geru( m_, n_, alpha, x, incx_, y, incy_, A, lda_ );
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda )
{
    impl::ger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda )
{
    impl::ger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//==============================================================================
// Unconjugated x y^T versions.

//------------------------------------------------------------------------------
/// CPU, float, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda )
{
    ger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda )
{
    ger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<float>, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<double>, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

}  // namespace blas
