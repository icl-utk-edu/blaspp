// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup rotg
void rotg(
    float *a,
    float *b,
    float *c,
    float *s )
{
    BLAS_srotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
/// @ingroup rotg
void rotg(
    double *a,
    double *b,
    double *c,
    double *s )
{
    BLAS_drotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
/// @ingroup rotg
void rotg(
    std::complex<float> *a,
    std::complex<float> *b,  // const in BLAS implementation, oddly
    float *c,
    std::complex<float> *s )
{
    BLAS_crotg( (blas_complex_float*) a,
                (blas_complex_float*) b,
                c,
                (blas_complex_float*) s );
}

// -----------------------------------------------------------------------------
/// @ingroup rotg
void rotg(
    std::complex<double> *a,
    std::complex<double> *b,  // const in BLAS implementation, oddly
    double *c,
    std::complex<double> *s )
{
    BLAS_zrotg( (blas_complex_double*) a,
                (blas_complex_double*) b,
                c,
                (blas_complex_double*) s );
}

}  // namespace blas
