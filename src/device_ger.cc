// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>
#include <string.h>

namespace blas {

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
    scalar_t*       A, int64_t lda,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    static_assert( is_complex_v<scalar_t>, "complex version" );

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
        counter::dev_ger_type element;
        memset( &element, 0, sizeof( element ) );
        element = { m, n };
        counter::insert( element, counter::Id::dev_ger );

        double gflops = 1e9 * blas::Gflop< scalar_t >::ger( m, n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    device_blas_int m_    = to_device_blas_int( m );
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int lda_  = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    if (layout == Layout::RowMajor) {
        // conjugate y (in y2)
        scalar_t* y2 = blas::device_malloc<scalar_t>( n, queue );
        blas::conj( n, y, incy, y2, 1, queue );
        incy_ = 1;
        queue.sync();

        // swap m <=> n, x <=> y, call geru
        internal::geru( n_, m_, alpha, y2, incy_, x, incx_, A, lda_, queue );
        queue.sync();

        blas::device_free( y2, queue );
    }
    else {
        internal::ger( m_, n_, alpha, x, incx_, y, incy_, A, lda_, queue );
    }
#endif
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
    scalar_t*       A, int64_t lda,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
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
        counter::dev_geru_type element;
        memset( &element, 0, sizeof( element ) );
        element = { m, n };
        counter::insert( element, counter::Id::dev_geru );

        double gflops = 1e9 * blas::Gflop< scalar_t >::ger( m, n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    device_blas_int m_    = to_device_blas_int( m );
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int lda_  = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    if (layout == Layout::RowMajor) {
        // swap m <=> n, x <=> y
        internal::geru( n_, m_, alpha, y, incy_, x, incx_, A, lda_, queue );
    }
    else {
        internal::geru( m_, n_, alpha, x, incx_, y, incy_, A, lda_, queue );
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    float const  alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    double const  alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> const  alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::ger( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> const  alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::ger( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//==============================================================================
// Unconjugated x y^T versions.

//------------------------------------------------------------------------------
/// GPU device, float, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    float const  alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda,
    blas::Queue& queue )
{
    ger( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    double const  alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda,
    blas::Queue& queue )
{
    ger( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float>, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> const  alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double>, unconjugated x y^T version.
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> const  alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::geru( layout, m, n, alpha, x, incx, y, incy, A, lda, queue );
}

}  // namespace blas
