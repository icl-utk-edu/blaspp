// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <stdio.h>
#include <complex>

#include "config.h"

#define LAPACK_dpstrf_base FORTRAN_NAME( dpstrf, DPSTRF )

#ifdef __cplusplus
extern "C"
#endif
void LAPACK_dpstrf_base(
    const char* uplo, const lapack_int* n,
    double* A, const lapack_int* lda,
    lapack_int* ipiv, lapack_int* rank,
    const double* tol,
    double* work,
    lapack_int* info
    #ifdef LAPACK_FORTRAN_STRLEN_END
    , size_t uplo_len
    #endif
    );

#ifdef LAPACK_FORTRAN_STRLEN_END
    #define LAPACK_dpstrf( ... ) LAPACK_dpstrf_base( __VA_ARGS__, 1 )
#else
    #define LAPACK_dpstrf( ... ) LAPACK_dpstrf_base( __VA_ARGS__ )
#endif

//------------------------------------------------------------------------------
int main()
{
    // If lapack_int is 32-bit, but LAPACK actually interprets it as 64-bit,
    // LAPACK will see n = 0x500000005 and segfault.
    // If lapack_int is 64-bit, LAPACK can interpret it as 32-bit or 64-bit
    // to see n = 5 and pass.
    lapack_int n[] = { 5, 5 };
    // symmetric positive definite A = L L^T.
    // -1 values in upper triangle (viewed column-major) are not referenced.
    double A[] = {
        4,  2,  0,  0,  0,
       -1,  5,  2,  0,  0,
       -1, -1,  5,  2,  0,
       -1, -1, -1,  5,  2,
       -1, -1, -1, -1,  5
    };
    lapack_int ipiv[5] = { -1, -1, -1, -1, -1 };
    lapack_int rank = -1;
    double tol = -1;
    double work[2*5];
    lapack_int info = -1;
    // With pivoting in pstrf, P^T A P = L2 L2^T.
    // Don't have exact L2 for comparison.
    LAPACK_dpstrf( "lower", n, A, n, ipiv, &rank, &tol, work, &info );
    bool okay = (info == 0) && (rank == 5);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
