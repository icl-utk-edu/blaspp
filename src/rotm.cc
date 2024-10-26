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

// =============================================================================
// Overloaded wrappers for s, d precisions.
// Not available for complex.

// -----------------------------------------------------------------------------
/// @ingroup rotm
void rotm(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy,
    float const param[5] )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rotm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rotm );

        double gflops = 1e9 * blas::Gflop< float >::rotm( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );
    BLAS_srotm( &n_, x, &incx_, y, &incy_, param );
}

// -----------------------------------------------------------------------------
/// @ingroup rotm
void rotm(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy,
    double const param[5] )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rotm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { n };
        counter::insert( element, counter::Id::rotm );

        double gflops = 1e9 * blas::Gflop< double >::rotm( n );
        counter::inc_flop_count( (long long int)gflops );
    #endif

    // convert arguments
    blas_int n_    = to_blas_int( n );
    blas_int incx_ = to_blas_int( incx );
    blas_int incy_ = to_blas_int( incy );
    BLAS_drotm( &n_, x, &incx_, y, &incy_, param );
}

}  // namespace blas
