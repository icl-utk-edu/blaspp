// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>

#include "config.h"

//------------------------------------------------------------------------------
int main()
{
    _Float16 a = 0.1;
    _Float16 b = 0.2;
    _Float16 c = a + b;

    printf( "%f + %f = %f -- expected 0.3\n", (float)a, (float)b, (float)c );
    return 0;
}
