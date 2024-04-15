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
/// @ingroup scal_internal
inline void scal(
    blas_int n,
    float alpha,
    float* x, blas_int incx )
{
    BLAS_sscal( &n, &alpha, x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup scal_internal
inline void scal(
    blas_int n,
    double alpha,
    double* x, blas_int incx )
{
    BLAS_dscal( &n, &alpha, x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup scal_internal
inline void scal(
    blas_int n,
    std::complex<float> alpha,
    std::complex<float>* x, blas_int incx )
{
    BLAS_cscal( &n,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup scal_internal
inline void scal(
    blas_int n,
    std::complex<double> alpha,
    std::complex<double>* x, blas_int incx )
{
    BLAS_zscal( &n,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup scal_internal
///
template <typename scalar_t>
void scal(
    int64_t n,
    scalar_t alpha,
    scalar_t* x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::scal_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::scal );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );

    // call low-level wrapper
    internal::scal( n_, alpha, x, incx_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup scal
void scal(
    int64_t n,
    float alpha,
    float* x, int64_t incx )
{
    impl::scal( n, alpha, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup scal
void scal(
    int64_t n,
    double alpha,
    double* x, int64_t incx )
{
    impl::scal( n, alpha, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup scal
void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float>* x, int64_t incx )
{
    impl::scal( n, alpha, x, incx );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup scal
void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double>* x, int64_t incx )
{
    impl::scal( n, alpha, x, incx );
}

}  // namespace blas
