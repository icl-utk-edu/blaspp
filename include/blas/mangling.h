// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

/// @file
/// Fortran name mangling macros for BLAS function calls.
///
/// This file defines macros for proper name mangling when calling Fortran BLAS
/// libraries from C/C++. Different Fortran compilers use different name mangling
/// conventions:
///
/// - Most compilers (GNU, Intel, PGI, Cray): lowercase with underscore (e.g., dgemm_)
/// - IBM xlf: lowercase without underscore (e.g., dgemm)
/// - Some compilers: uppercase (e.g., DGEMM)
///
/// The BLAS_FORTRAN_NAME macro automatically selects the correct mangling based on
/// compiler-defined macros BLAS_FORTRAN_UPPER or BLAS_FORTRAN_LOWER.
///
/// @note This file is included by fortran.h which declares all Fortran BLAS prototypes.

#ifndef BLAS_MANGLING_H
#define BLAS_MANGLING_H

#include "blas/defines.h"

// -----------------------------------------------------------------------------
// Fortran name mangling depends on compiler.
// Define FORTRAN_UPPER for uppercase,
// define FORTRAN_LOWER for lowercase (IBM xlf),
// else the default is lowercase with appended underscore
// (GNU gcc, Intel icc, PGI pgfortan, Cray ftn).
#ifndef BLAS_FORTRAN_NAME
    #if defined(BLAS_FORTRAN_UPPER)
        #define BLAS_FORTRAN_NAME( lower, UPPER ) UPPER
    #elif defined(BLAS_FORTRAN_LOWER)
        #define BLAS_FORTRAN_NAME( lower, UPPER ) lower
    #else
        #define BLAS_FORTRAN_NAME( lower, UPPER ) lower##_
    #endif
#endif

#endif        //  #ifndef BLAS_MANGLING_H
