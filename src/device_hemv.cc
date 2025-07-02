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
/// @ingroup hemv_internal
///
template <typename scalar_t>
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* x, int64_t incx,
    scalar_t beta,
    scalar_t*       y, int64_t incy,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    static_assert( is_complex_v<scalar_t>, "complex version" );

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
        counter::dev_hemv_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::dev_hemv );

        double gflops = 1e9 * blas::Gflop< scalar_t >::hemv( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int lda_  = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );
    device_blas_int incy_ = to_device_blas_int( incy );

    blas::internal_set_device( queue.device() );

    // Deal with layout. RowMajor needs copy of x in x2;
    // in other cases, x2 == x.
    scalar_t* x2 = const_cast< scalar_t* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate alpha, beta, x (in x2), and y (in-place)
        alpha = conj( alpha );
        beta  = conj( beta );

        x2 = blas::device_malloc<scalar_t>( n, queue );
        blas::conj( n, x, incx, x2, 1, queue );
        incx_ = 1;

        blas::conj( n, y, incy, y, incy, queue );
    }
    queue.sync();

    // call low-level wrapper
    internal::hemv( uplo, n_,
                    alpha, A, lda_, x2, incx_, beta, y, incy_, queue );

    if (layout == Layout::RowMajor) {
        // y = conj( y )
        blas::conj( n, y, incy, y, incy, queue );
        queue.sync();
        blas::device_free( x2, queue );
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU, float version.
/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float*       y, int64_t incy,
    blas::Queue& queue )
{
    symv( layout, uplo, n,
          alpha, A, lda, x, incx, beta, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU, double version.
/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double*       y, int64_t incy,
    blas::Queue& queue )
{
    symv( layout, uplo, n,
          alpha, A, lda, x, incx, beta, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU, complex<float> version.
/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::hemv( layout, uplo, n,
                alpha, A, lda, x, incx, beta, y, incy, queue );
}

//------------------------------------------------------------------------------
/// GPU, complex<double> version.
/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::hemv( layout, uplo, n,
                alpha, A, lda, x, incx, beta, y, incy, queue );
}

}  // namespace blas
