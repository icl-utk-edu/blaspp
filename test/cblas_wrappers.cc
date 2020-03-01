// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Function definitions moved from cblas.hh for ESSL compatability.

#include "blas/fortran.h"

#include <complex>

void
cblas_rotg(
    std::complex<float> *a, std::complex<float> *b,
    float *c, std::complex<float> *s )
{
    BLAS_crotg(
        (blas_complex_float*) a,
        (blas_complex_float*) b,
        c,
        (blas_complex_float*) s );
}

void
cblas_rotg(
    std::complex<double> *a, std::complex<double> *b,
    double *c, std::complex<double> *s )
{
    BLAS_zrotg(
        (blas_complex_double*) a,
        (blas_complex_double*) b,
        c,
        (blas_complex_double*) s );
}
