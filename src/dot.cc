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
/// @ingroup dot_internal
inline float dot(
    blas_int n,
    float const* x, blas_int incx,
    float const* y, blas_int incy )
{
    return BLAS_sdot( &n, x, &incx, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup dot_internal
inline double dot(
    blas_int n,
    double const* x, blas_int incx,
    double const* y, blas_int incy )
{
    return BLAS_ddot( &n, x, &incx, y, &incy );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup dot_internal
inline std::complex<float> dot(
    blas_int n,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> const* y, blas_int incy )
{
    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        BLAS_cdotc( (blas_complex_float*) &value, &n,
                    (blas_complex_float*) x, &incx,
                    (blas_complex_float*) y, &incy );
        return value;
    #else
        // GNU gcc convention
        blas_complex_float value
             = BLAS_cdotc( &n,
                           (blas_complex_float*) x, &incx,
                           (blas_complex_float*) y, &incy );
        return *reinterpret_cast< std::complex<float>* >( &value );
    #endif
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup dot_internal
inline std::complex<double> dot(
    blas_int n,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> const* y, blas_int incy )
{
    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        BLAS_zdotc( (blas_complex_double*) &value, &n,
                    (blas_complex_double*) x, &incx,
                    (blas_complex_double*) y, &incy );
        return value;
    #else
        // GNU gcc convention
        blas_complex_double value
             = BLAS_zdotc( &n,
                           (blas_complex_double*) x, &incx,
                           (blas_complex_double*) y, &incy );
        return *reinterpret_cast< std::complex<double>* >( &value );
    #endif
}

//==============================================================================
// Unconjugated x^T y versions.

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float>, unconjugated x^T y
/// version.
/// @ingroup dotu_internal
inline std::complex<float> dotu(
    blas_int n,
    std::complex<float> const* x, blas_int incx,
    std::complex<float> const* y, blas_int incy )
{
    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        BLAS_cdotu( (blas_complex_float*) &value, &n,
                    (blas_complex_float*) x, &incx,
                    (blas_complex_float*) y, &incy );
        return value;
    #else
        // GNU gcc convention
        blas_complex_float value
             = BLAS_cdotu( &n,
                           (blas_complex_float*) x, &incx,
                           (blas_complex_float*) y, &incy );
        return *reinterpret_cast< std::complex<float>* >( &value );
    #endif
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double>, unconjugated x^T y
/// version.
/// @ingroup dotu_internal
inline std::complex<double> dotu(
    blas_int n,
    std::complex<double> const* x, blas_int incx,
    std::complex<double> const* y, blas_int incy )
{
    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        BLAS_zdotu( (blas_complex_double*) &value, &n,
                    (blas_complex_double*) x, &incx,
                    (blas_complex_double*) y, &incy );
        return value;
    #else
        // GNU gcc convention
        blas_complex_double value
             = BLAS_zdotu( &n,
                           (blas_complex_double*) x, &incx,
                           (blas_complex_double*) y, &incy );
        return *reinterpret_cast< std::complex<double>* >( &value );
    #endif
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// Conjugated x^H y version.
/// @ingroup dot_internal
///
template <typename scalar_t>
scalar_t dot(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dot );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    // call low-level wrapper
    return internal::dot( n_, x, incx_, y, incy_ );
}

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// Unconjugated x^T y version.
/// @ingroup dotu_internal
///
template <typename scalar_t>
scalar_t dotu(
    int64_t n,
    scalar_t const* x, int64_t incx,
    scalar_t const* y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::dotu_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::dotu );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );

    // call low-level wrapper
    return internal::dotu( n_, x, incx_, y, incy_ );
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup dot
float dot(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy )
{
    return impl::dot( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup dot
double dot(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy )
{
    return impl::dot( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup dot
std::complex<float> dot(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy )
{
    return impl::dot( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup dot
std::complex<double> dot(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy )
{
    return impl::dot( n, x, incx, y, incy );
}

//==============================================================================
// Unconjugated x y^T versions.

//------------------------------------------------------------------------------
/// CPU, float, unconjugated x^T y version.
/// @ingroup dotu
float dotu(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, double, unconjugated x^T y version.
/// @ingroup dotu
double dotu(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<float>, unconjugated x^T y version.
/// @ingroup dotu
std::complex<float> dotu(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy )
{
    return impl::dotu( n, x, incx, y, incy );
}

//------------------------------------------------------------------------------
/// CPU, complex<double>, unconjugated x^T y version.
/// @ingroup dotu
std::complex<double> dotu(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy )
{
    return impl::dotu( n, x, incx, y, incy );
}

}  // namespace blas
