// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>

namespace blas {

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
    scalar_t*       A, int64_t lda,
    blas::Queue& queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
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
        counter::dev_her_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, n };
        counter::insert( element, counter::Id::dev_her );

        double gflops = 1e9 * blas::Gflop< scalar_t >::her( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int lda_  = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::internal_set_device( queue.device() );

    // Deal with layout. RowMajor needs copy of x in x2;
    // in other cases, x2 == x.
    scalar_t* x2 = const_cast< scalar_t* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate x (in x2)
        x2 = blas::device_malloc<scalar_t>( n, queue );
        blas::conj( n, x, incx, x2, 1, queue );
        incx_ = 1;
    }
    queue.sync();

    // call low-level wrapper
    internal::her( uplo, n_,
                   alpha, x2, incx_, A, lda_, queue );

    if (layout == Layout::RowMajor) {
        device_free( x2, queue );
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float*       A, int64_t lda,
    blas::Queue& queue )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double*       A, int64_t lda,
    blas::Queue& queue )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::her( layout, uplo, n,
               alpha, x, incx, A, lda, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>*       A, int64_t lda,
    blas::Queue& queue )
{
    impl::her( layout, uplo, n,
               alpha, x, incx, A, lda, queue );
}

}  // namespace blas
