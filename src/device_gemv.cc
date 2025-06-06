// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"
#include "blas/counter.hh"

#include "device_internal.hh"

#include <limits>
#include "blas/counter.hh"

namespace blas {

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup gemv_internal
///
template <typename scalar_t>
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
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
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor) {
        if (trans == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < n );
    }
    else {
        if (trans != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < n );
    }

    // convert arguments
    device_blas_int m_ = to_device_blas_int( m );
    device_blas_int n_ = to_device_blas_int( n );
    device_blas_int lda_ = to_device_blas_int( lda );
    device_blas_int incx_ = to_device_blas_int( incx );

    blas::internal_set_device( queue.device() );
    
    // call low-level wrapper
    if (layout == Layout::RowMajor) {
        // swap m <=> n
        internal::gemv( trans, n_, m_, alpha, A, lda_,
                        x, incx_, beta, y, incy, queue );
    }
    else {
        internal::gemv( trans, m_, n_, alpha, A, lda_,
                        x, incx_, beta, y, incy, queue );
    }
    
#endif
}

} // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// GPU device, float version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue);
}

//------------------------------------------------------------------------------
/// GPU device, double version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue);
}

//------------------------------------------------------------------------------
/// GPU device, complex<float> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue);
}

//------------------------------------------------------------------------------
/// GPU device, complex<double> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>*       y, int64_t incy,
    blas::Queue& queue )
{
    impl::gemv( layout, trans, m, n, alpha, A, lda,
                x, incx, beta, y, incy, queue);
}

} // namespace blas