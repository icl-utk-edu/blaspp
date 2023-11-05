// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
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
