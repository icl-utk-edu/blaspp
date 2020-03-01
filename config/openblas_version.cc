// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <openblas_config.h>

int main()
{
    const char* v = OPENBLAS_VERSION;
    printf( "OPENBLAS_VERSION=%s\n", v );
    return 0;
}
