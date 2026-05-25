// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_MANGLING_H
#define BLAS_MANGLING_H

#include "blas/defines.h"

// -----------------------------------------------------------------------------
// Fortran name mangling depends on compiler.
// Define FORTRAN_UPPER for uppercase,
// define FORTRAN_LOWER for lowercase (IBM xlf),
// else the default is lowercase with appended underscore
// (GNU gcc, Intel icc, PGI pgfortan, Cray ftn).
//
// BLAS_FORTRAN_SUFFIX, if defined, is appended to every mangled symbol.
// Used to link against BLAS libraries that suffix their symbols, e.g.,
// OpenBLAS built with INTERFACE64=1 SYMBOLSUFFIX=64_ exports `dgemm_64_`.
// Pass via CMake's `blas_symbol_suffix` option.
#ifndef BLAS_FORTRAN_NAME
    #ifdef BLAS_FORTRAN_SUFFIX
        #define BLAS_FORTRAN_CONCAT_( a, b ) a##b
        #define BLAS_FORTRAN_CONCAT(  a, b ) BLAS_FORTRAN_CONCAT_( a, b )
        #if defined(BLAS_FORTRAN_UPPER)
            #define BLAS_FORTRAN_NAME( lower, UPPER ) \
                BLAS_FORTRAN_CONCAT( UPPER, BLAS_FORTRAN_SUFFIX )
        #elif defined(BLAS_FORTRAN_LOWER)
            #define BLAS_FORTRAN_NAME( lower, UPPER ) \
                BLAS_FORTRAN_CONCAT( lower, BLAS_FORTRAN_SUFFIX )
        #else
            #define BLAS_FORTRAN_NAME( lower, UPPER ) \
                BLAS_FORTRAN_CONCAT( lower##_, BLAS_FORTRAN_SUFFIX )
        #endif
    #else
        #if defined(BLAS_FORTRAN_UPPER)
            #define BLAS_FORTRAN_NAME( lower, UPPER ) UPPER
        #elif defined(BLAS_FORTRAN_LOWER)
            #define BLAS_FORTRAN_NAME( lower, UPPER ) lower
        #else
            #define BLAS_FORTRAN_NAME( lower, UPPER ) lower##_
        #endif
    #endif
#endif

#endif        //  #ifndef BLAS_MANGLING_H
