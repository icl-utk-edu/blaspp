// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <omp.h>
#include <stdio.h>

int main()
{
    int nthreads = 1;
    int tid = 0;
    #pragma omp parallel
    {
        nthreads = omp_get_max_threads();
        tid = omp_get_thread_num();
        printf( "tid %d, nthreads %d\n", tid, nthreads );
    }
    printf( "ok\n" );
    return 0;
}
