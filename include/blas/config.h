// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_CONFIG_H
#define BLAS_CONFIG_H

#include <stdint.h>

#include "blas/defines.h"

#ifndef blas_int
    #if defined( BLAS_ILP64 ) && defined( ACCELERATE_NEW_LAPACK )
        #ifndef ACCELERATE_LAPACK_ILP64
        #define ACCELERATE_LAPACK_ILP64
        #endif
        typedef long blas_int;
    #elif defined( BLAS_ILP64 )
        typedef int64_t blas_int;
    #else
        typedef int blas_int;
    #endif
    /* #define so that #ifdef works. */
    #define blas_int blas_int
#endif

/* f2c, hence MacOS Accelerate, returns double instead of float
 * for sdot, slange, clange, etc. */
#if defined(BLAS_HAVE_F2C)
    typedef double blas_float_return;
#else
    typedef float blas_float_return;
#endif

#if defined(BLAS_COMPLEX_CPP) || defined(LAPACK_COMPLEX_CPP)
    /* user has to specifically request std::complex,
     * as it isn't compatible as a return type from extern C functions. */
    #include <complex>
    typedef std::complex<float>  blas_complex_float;
    typedef std::complex<double> blas_complex_double;
#elif defined(_MSC_VER)
    /* MSVC has no C99 _Complex */
    typedef struct { float real, imag; }  blas_complex_float;
    typedef struct { double real, imag; } blas_complex_double;
#else
    /* otherwise, by default use C99 _Complex */
    #include <complex.h>
    typedef float _Complex  blas_complex_float;
    typedef double _Complex blas_complex_double;
#endif

/* define so we can check later with ifdef */
#define blas_complex_float  blas_complex_float
#define blas_complex_double blas_complex_double

#endif        //  #ifndef BLAS_CONFIG_H
