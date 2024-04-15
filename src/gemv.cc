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
/// @ingroup gemv_internal
inline void gemv(
    char trans,
    blas_int m, blas_int n,
    float alpha,
    float const* A, blas_int lda,
    float const* x, blas_int incx,
    float beta,
    float*       y, blas_int incy )
{
    BLAS_sgemv( &trans, &m, &n,
                &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup gemv_internal
inline void gemv(
    char trans,
    blas_int m, blas_int n,
    double alpha,
    double const* A, blas_int lda,
    double const* x, blas_int incx,
    double beta,
    double*       y, blas_int incy )
{
    BLAS_dgemv( &trans, &m, &n,
                &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup gemv_internal
inline void gemv(
    char trans,
    blas_int m, blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* A, blas_int lda,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> beta,
    std::complex<float>*       y, blas_int incy )
{
    BLAS_cgemv( &trans, &m, &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda,
                (blas_complex_float*) x, &incx,
                (blas_complex_float*) &beta,
                (blas_complex_float*) y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup gemv_internal
inline void gemv(
    char trans,
    blas_int m, blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* A, blas_int lda,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> beta,
    std::complex<double>*       y, blas_int incy )
{
    BLAS_zgemv( &trans, &m, &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda,
                (blas_complex_double*) x, &incx,
                (blas_complex_double*) &beta,
                (blas_complex_double*) y, &incy );
}

}  // namespace internal

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
    scalar_t*       y, int64_t incy )
{
    using std::swap;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::gemv_type element;
        memset( &element, 0, sizeof( element ) );
        element = { trans, m, n };
        counter::insert( element, counter::Id::gemv );
    #endif

    // convert arguments
    blas_int m_    = to_blas_int( m );
    blas_int n_    = to_blas_int( n );
    blas_int lda_  = to_blas_int( lda );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    // Deal with layout. RowMajor ConjTrans needs copy of x in x2;
    // in other cases, x2 == x.
    scalar_t* x2 = const_cast< scalar_t* >( x );
    Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        if constexpr (is_complex<scalar_t>::value) {
            if (trans == Op::ConjTrans) {
                // conjugate alpha, beta, x (in x2), and y (in-place)
                alpha = conj( alpha );
                beta  = conj( beta );

                x2 = new scalar_t[ m ];
                int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
                for (int64_t i = 0; i < m; ++i) {
                    x2[ i ] = conj( x[ ix ] );
                    ix += incx;
                }
                incx_ = 1;

                int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
                for (int64_t i = 0; i < n; ++i) {
                    y[ iy ] = conj( y[ iy ] );
                    iy += incy;
                }
            }
        }
        // A => A^T; A^T => A; A^H => A + conj
        swap( m_, n_ );
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }
    char trans_ = op2char( trans2 );

    // call low-level wrapper
    internal::gemv( trans_, m_, n_,
                    alpha, A, lda_, x2, incx_, beta, y, incy_ );

    if constexpr (is_complex<scalar_t>::value) {
        if (x2 != x) {  // RowMajor ConjTrans
            // y = conj( y )
            int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t i = 0; i < n; ++i) {
                y[ iy ] = conj( y[ iy ] );
                iy += incy;
            }
            delete[] x2;
        }
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float*       y, int64_t incy )
{
    impl::gemv( layout, trans, m, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double*       y, int64_t incy )
{
    impl::gemv( layout, trans, m, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>*       y, int64_t incy )
{
    impl::gemv( layout, trans, m, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>*       y, int64_t incy )
{
    impl::gemv( layout, trans, m, n,
                alpha, A, lda, x, incx, beta, y, incy );
}

}  // namespace blas
