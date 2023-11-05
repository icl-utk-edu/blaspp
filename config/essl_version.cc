// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <essl.h>

int main()
{
    int v = iessl();
    int version      = int( v / 1000000 );
    int release      = int( (v % 1000000) / 10000 );
    int modification = int( (v % 10000) / 100 );
    int ptf          = v % 100;

    printf( "ESSL_VERSION=%d.%d.%d.%d\n",
            version, release, modification, ptf );
    return 0;
}
