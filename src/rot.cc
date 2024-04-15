// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas/counter.hh"

#include <limits>
#include <string.h>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup rot
void rot(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy,
    float c,
    float s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_srot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// -----------------------------------------------------------------------------
/// @ingroup rot
void rot(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy,
    double c,
    double s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_drot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// -----------------------------------------------------------------------------
/// @ingroup rot
/// real cosine, real sine
/// This variant, with real sine, is used in the SVD or EVD where a real
/// Givens rotation, that eliminates entries in the real bi- or
/// tridiagonal matrix, is applied to complex singular or eigen vectors.
///
void rot(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy,
    float c,
    float s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_csrot( &n_,
                (blas_complex_float*) x, &incx_,
                (blas_complex_float*) y, &incy_,
                &c, &s );
}

// -----------------------------------------------------------------------------
/// @ingroup rot
/// real cosine, real sine
void rot(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy,
    double c,
    double s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_zdrot( &n_,
                (blas_complex_double*) x, &incx_,
                (blas_complex_double*) y, &incy_,
                &c, &s );
}

// -----------------------------------------------------------------------------
/// @ingroup rot
/// real cosine, complex sine
/// This variant, with complex sine, is used to eliminate entries in a
/// complex matrix.
///
void rot(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy,
    float c,
    std::complex<float> s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_crot( &n_,
               (blas_complex_float*) x, &incx_,
               (blas_complex_float*) y, &incy_,
               &c,
               (blas_complex_float*) &s );
}

// -----------------------------------------------------------------------------
/// @ingroup rot
/// real cosine, complex sine
void rot(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy,
    double c,
    std::complex<double> s )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rot_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rot );
    #endif

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_zrot( &n_,
               (blas_complex_double*) x, &incx_,
               (blas_complex_double*) y, &incy_,
               &c,
               (blas_complex_double*) &s );
}

}  // namespace blas
