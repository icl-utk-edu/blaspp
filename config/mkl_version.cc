// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <mkl.h>

int main()
{
    MKLVersion v;
    MKL_Get_Version( &v );
    printf( "MKL_VERSION=%d.%d.%d\n",
            v.MajorVersion, v.MinorVersion, v.UpdateVersion );
    return 0;
}
