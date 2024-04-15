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
    scalar_t*       C, int64_t ldc,
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
        counter::dev_herk_type element;
        memset( &element, 0, sizeof( element ) );
        element = { uplo, trans, n, k };
        counter::insert( element, counter::Id::dev_herk );
    #endif

    // convert arguments
    device_blas_int n_   = to_device_blas_int( n );
    device_blas_int k_   = to_device_blas_int( k );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int ldc_ = to_device_blas_int( ldc );

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^H; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);
    }

    blas::internal_set_device( queue.device() );

    // call low-level wrapper
    internal::herk( uplo, trans, n_, k_,
                    alpha, A, lda_, beta, C, ldc_, queue );
#endif
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue )
{
    blas::syrk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue )
{
    blas::syrk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,  // note: real
    std::complex<float> const* A, int64_t lda,
    float beta,   // note: real
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc, queue );
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    std::complex<double> const* A, int64_t lda,
    double beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue )
{
    impl::herk( layout, uplo, trans, n, k,
                alpha, A, lda, beta, C, ldc, queue );
}

}  // namespace blas
