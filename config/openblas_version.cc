// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

// OpenBLAS built with SYMBOLSUFFIX (e.g., 64_) also suffixes its own
// functions, exporting openblas_get_config64_. Defined before including
// cblas.h, this works whether or not the header's declarations are suffixed.
#ifdef BLAS_FORTRAN_SUFFIX
    #define OPENBLAS_CONCAT_( a, b ) a##b
    #define OPENBLAS_CONCAT(  a, b ) OPENBLAS_CONCAT_( a, b )
    #define openblas_get_config \
        OPENBLAS_CONCAT( openblas_get_config, BLAS_FORTRAN_SUFFIX )
#endif

#include <cblas.h> // openblas_get_config

int main()
{
    const char* v = OPENBLAS_VERSION;
    printf( "OPENBLAS_VERSION=%s\n", v );

    // since OPENBLAS_VERSION is defined in the header, it may work even
    // if we don't link with openblas. Calling an OpenBLAS-specific
    // function ensures we are linking with OpenBLAS.
    const char* config = openblas_get_config();
    printf( "openblas_get_config=%s\n", config );

    return 0;
}
