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
// Not available for complex.

// -----------------------------------------------------------------------------
/// @ingroup rotmg
void rotmg(
    float *d1,
    float *d2,
    float *a,
    float  b,
    float  param[5] )
{
    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rotmg_type element;
        memset( &element, 0, sizeof( element ) );
        element = { 1 };
        counter::insert( element, counter::Id::rotmg );
    #endif

    BLAS_srotmg( d1, d2, a, &b, param );
}

// -----------------------------------------------------------------------------
/// @ingroup rotmg
void rotmg(
    double *d1,
    double *d2,
    double *a,
    double  b,
    double  param[5] )
{
    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::rotmg_type element;
        memset( &element, 0, sizeof( element ) );
        element = { 1 };
        counter::insert( element, counter::Id::rotmg );
    #endif

    BLAS_drotmg( d1, d2, a, &b, param );
}

}  // namespace blas
