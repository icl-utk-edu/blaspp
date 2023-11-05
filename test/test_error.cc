// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"
#include "blas/util.hh"

void test_error( Params& params, bool run )
{
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();

    if (! run) {
        printf( "test error checks the internal error routines:\n"
                "if m == 100, checks: blas_error_if( m == n );\n"
                "if m == 200, checks: blas_error_if_msg( m == n, \"m %%d == n %%d\", m, n );\n"
                "if m == 300, checks: assert( m != n );\n\n" );
        return;
    }

    if (m == 100) {
        blas_error_if( m == n );
    }
    else if (m == 200) {
        blas_error_if_msg( m == n, "m %lld == n %lld", llong( m ), llong( n ) );
    }
    else if (m == 300) {
        assert( m != n );
    }
}
