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
/// @ingroup trsv_internal
///
template <typename scalar_t>
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    scalar_t const* A, int64_t lda,
    scalar_t*       x, int64_t incx,
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
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dev_trsv_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, trans, diag, n };
        counter::insert( element, counter::Id::dev_trsv );

        double gflops = 1e9 * blas::Gflop< scalar_t >::trsv( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    device_blas_int n_    = to_device_blas_int( n );
    device_blas_int lda_  = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::internal_set_device( queue.device() );

    blas::Uplo uplo_  = uplo;
    blas::Op   trans_ = trans;
    blas::Diag diag_  = diag;
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo_ = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans_ = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);

        if constexpr (is_complex_v<scalar_t>) {
            if (trans == Op::ConjTrans) {
                // conjugate x (in-place)
                blas::conj( n, x, incx, x, incx, queue );
            }
        }
    }
    queue.sync();

    // call low-level wrapper
    internal::trsv( uplo_, trans_, diag_, n_, A, lda_, x, incx_, queue );

    if constexpr (is_complex_v<scalar_t>) {
        if (layout == Layout::RowMajor && trans == Op::ConjTrans) {
            // conjugate x (in-place)
            blas::conj( n, x, incx, x, incx, queue );
        }
    }
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const* A, int64_t lda,
    float*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const* A, int64_t lda,
    double*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       x, int64_t incx,
    blas::Queue& queue )
{
    impl::trsv( layout, uplo, trans, diag, n, A, lda, x, incx, queue );
}

}  // namespace blas
