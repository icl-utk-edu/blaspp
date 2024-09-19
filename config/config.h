// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CONFIG_H
#define CONFIG_H

#include <stdint.h>

//------------------------------------------------------------------------------
#if defined(FORTRAN_UPPER)
    #pragma message "Fortran upper"
    #define FORTRAN_NAME( lower, UPPER ) UPPER
#elif defined(FORTRAN_LOWER)
    #pragma message "Fortran lower"
    #define FORTRAN_NAME( lower, UPPER ) lower
#else
    // default is ADD_
    #define FORTRAN_NAME( lower, UPPER ) lower ## _
#endif

//------------------------------------------------------------------------------
#if defined(BLAS_ILP64) || defined(LAPACK_ILP64)
    // long is >= 32 bits, long long is >= 64 bits
    // macOS Accelerate uses long, Intel MKL uses long long,
    // prefer int64_t (which can be long or long long).
    #ifdef BLAS_HAVE_ACCELERATE
        #pragma message "Accelerate ilp64 (long)"
        #define ACCELERATE_LAPACK_ILP64
        typedef long blas_int;
        typedef long lapack_int;
    #else
        #pragma message "ilp64 (int64_t)"
        typedef int64_t blas_int;
        typedef int64_t lapack_int;
    #endif
#else
    typedef int blas_int;
    typedef int lapack_int;
#endif

//------------------------------------------------------------------------------
#ifdef BLAS_HAVE_ACCELERATE
    // Neither old nor new macOS Accelerate API passes strlen.
    #pragma message "Accelerate undef strlen"
    #undef BLAS_FORTRAN_STRLEN_END
    #undef LAPACK_FORTRAN_STRLEN_END
#else
    #ifndef BLAS_FORTRAN_STRLEN_END
    #define BLAS_FORTRAN_STRLEN_END
    #endif

    #ifndef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_FORTRAN_STRLEN_END
    #endif
#endif

#endif // CONFIG_H
