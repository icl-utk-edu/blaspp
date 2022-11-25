// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup syr2_internal
inline void syr2(
    char uplo,
    blas_int n,
    float alpha,
    float const* x, blas_int incx,
    float const* y, blas_int incy,
    float*       A, blas_int lda )
{
    BLAS_ssyr2( &uplo, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup syr2_internal
inline void syr2(
    char uplo,
    blas_int n,
    double alpha,
    double const* x, blas_int incx,
    double const* y, blas_int incy,
    double*       A, blas_int lda )
{
    BLAS_dsyr2( &uplo, &n, &alpha, x, &incx, y, &incy, A, &lda );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup syr2k_internal
/// todo: repeated from syr2k.cc
inline void syr2k(
    char uplo,
    char trans,
    blas_int n, blas_int k,
    std::complex<float> alpha,
    std::complex<float> const* A, blas_int lda,
    std::complex<float> const* B, blas_int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, blas_int ldc )
{
    BLAS_csyr2k( &uplo, &trans, &n, &k,
                 (blas_complex_float*) &alpha,
                 (blas_complex_float*) A, &lda,
                 (blas_complex_float*) B, &ldb,
                 (blas_complex_float*) &beta,
                 (blas_complex_float*) C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup syr2k_internal
/// todo: repeated from syr2k.cc
inline void syr2k(
    char uplo,
    char trans,
    blas_int n, blas_int k,
    std::complex<double> alpha,
    std::complex<double> const* A, blas_int lda,
    std::complex<double> const* B, blas_int ldb,
    std::complex<double> beta,
    std::complex<double>*       C, blas_int ldc )
{
    BLAS_zsyr2k( &uplo, &trans, &n, &k,
                 (blas_complex_double*) &alpha,
                 (blas_complex_double*) A, &lda,
                 (blas_complex_double*) B, &ldb,
                 (blas_complex_double*) &beta,
                 (blas_complex_double*) C, &ldc );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// LAPACK doesn't have [cz]syr2; this calls [cz]syr2k with k == 1, beta = 1.
/// @ingroup syr2_internal
///
template <typename scalar_t>
void syr2(
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

    if constexpr (! is_complex<scalar_t>::value) {
        // call low-level wrapper
        internal::syr2( uplo_, n_, alpha, x, incx_, y, incy_, A, lda_ );
    }
    else { // is_complex<scalar_t>
        // if x2=x and y2=y, then they aren't modified
        scalar_t* x2 = const_cast< scalar_t* >( x );
        scalar_t* y2 = const_cast< scalar_t* >( y );

        // no [cz]syr2 in BLAS or LAPACK, so use [cz]syr2k with k=1 and beta=1.
        // if   inc == 1, consider x and y as n-by-1 matrices in n-by-1 arrays,
        // elif inc >= 1, consider x and y as 1-by-n matrices in inc-by-n arrays,
        // else, copy x and y and use case inc == 1 above.
        blas_int k_ = 1;
        char trans_;
        blas_int ldx_, ldy_;
        if (incx == 1 && incy == 1) {
            trans_ = 'N';
            ldx_ = n_;
            ldy_ = n_;
        }
        else if (incx >= 1 && incy >= 1) {
            trans_ = 'T';
            ldx_ = incx_;
            ldy_ = incy_;
        }
        else {
            x2 = new scalar_t[ n ];
            y2 = new scalar_t[ n ];
            int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
            int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t i = 0; i < n; ++i) {
                x2[ i ] = x[ ix ];
                y2[ i ] = y[ iy ];
                ix += incx;
                iy += incy;
            }
            trans_ = 'N';
            ldx_ = n_;
            ldy_ = n_;
        }
        scalar_t beta = 1;

        // call low-level wrapper
        internal::syr2k( uplo_, trans_, n_, k_,
                         alpha, x2, ldx_, y2, ldy_, beta, A, lda_ );

        if (x2 != x) {
            delete[] x2;
            delete[] y2;
        }
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float*       A, int64_t lda )
{
    impl::syr2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double*       A, int64_t lda )
{
    impl::syr2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// LAPACK doesn't have [cz]syr2; this calls [cz]syr2k with k == 1, beta = 1.
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>*       A, int64_t lda )
{
    impl::syr2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// LAPACK doesn't have [cz]syr2; this calls [cz]syr2k with k == 1, beta = 1.
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>*       A, int64_t lda )
{
    impl::syr2( layout, uplo, n,
                alpha, x, incx, y, incy, A, lda );
}

}  // namespace blas
