// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CHECK_GEMM_HH
#define CHECK_GEMM_HH

#include "blas/util.hh"

// Test headers.
#include "lapack_wrappers.hh"

#include <limits>

// -----------------------------------------------------------------------------
// Computes error for multiplication with general matrix result.
// Covers dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?.
// Cnorm is norm of original C, before multiplication operation.
template <typename T>
void check_gemm(
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    bool verbose,
    blas::real_type<T> error[1],
    bool* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    using std::sqrt;
    using std::abs;
    using real_t = blas::real_type<T>;

    assert( m >= 0 );
    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= m );
    assert( ldcref >= m );

    // C -= Cref
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            C(i, j) -= Cref(i, j);
        }
    }

    real_t work[1], Cout_norm;
    Cout_norm = lapack_lange( "f", m, n, C, ldc, work );
    error[0] = Cout_norm
             / (sqrt( real_t( k ) + 2 ) * abs( alpha ) * Anorm * Bnorm
                 + 2 * abs( beta ) * Cnorm);
    if (verbose) {
        printf( "error: ||Cout||=%.2e / (sqrt(k=%lld + 2)"
                " * |alpha|=%.2e * ||A||=%.2e * ||B||=%.2e"
                " + 2 * |beta|=%.2e * ||C||=%.2e) = %.2e\n",
                Cout_norm, llong( k ),
                abs( alpha ), Anorm, Bnorm,
                abs( beta ), Cnorm, error[0] );
    }

    // complex needs extra factor; see Higham, 2002, sec. 3.6.
    if (blas::is_complex<T>::value) {
        error[0] /= 2*sqrt(2);
    }

    real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < u);

    #undef C
    #undef Cref
}

// -----------------------------------------------------------------------------
// Computes error for multiplication with symmetric or Hermitian matrix result.
// Covers syr, syr2, syrk, syr2k, her, her2, herk, her2k.
// Cnorm is norm of original C, before multiplication operation.
//
// alpha and beta are either real or complex, depending on routine:
//          zher    zher2   zherk   zher2k  zsyr    zsyr2   zsyrk   zsyr2k
// alpha    real    complex real    complex complex complex complex complex
// beta     --      --      real    real    --      --      complex complex
// zsyr2 doesn't exist in standard BLAS or LAPACK.
template <typename TA, typename TB, typename T>
void check_herk(
    blas::Uplo uplo,
    int64_t n, int64_t k,
    TA alpha,
    TB beta,
    blas::real_type<T> Anorm,
    blas::real_type<T> Bnorm,
    blas::real_type<T> Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    bool verbose,
    blas::real_type<T> error[1],
    bool* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    using std::sqrt;
    using std::abs;
    typedef blas::real_type<T> real_t;

    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= n );
    assert( ldcref >= n );

    // C -= Cref
    if (uplo == blas::Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = j; i < n; ++i) {
                C(i, j) -= Cref(i, j);
            }
        }
    }
    else {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= j; ++i) {
                C(i, j) -= Cref(i, j);
            }
        }
    }

    // For a rank-2k update, this should be
    // sqrt(k+3) |alpha| (norm(A)*norm(B^T) + norm(B)*norm(A^T))
    //     + 3 |beta| norm(C)
    // However, so far using the same bound as rank-k works fine.
    real_t work[1], Cout_norm;
    Cout_norm = lapack_lanhe( "f", uplo2str(uplo), n, C, ldc, work );
    error[0] = Cout_norm
             / (sqrt( real_t( k ) + 2 ) * abs( alpha ) * Anorm * Bnorm
                 + 2 * abs( beta ) * Cnorm);
    if (verbose) {
        printf( "error: ||Cout||=%.2e / (sqrt(k=%lld + 2)"
                " * |alpha|=%.2e * ||A||=%.2e * ||B||=%.2e"
                " + 2 * |beta|=%.2e * ||C||=%.2e) = %.2e\n",
                Cout_norm, llong( k ),
                abs( alpha ), Anorm, Bnorm,
                abs( beta ), Cnorm, error[0] );
    }

    // complex needs extra factor; see Higham, 2002, sec. 3.6.
    if (blas::is_complex<T>::value) {
        error[0] /= 2*sqrt(2);
    }

    real_t u = 0.5 * std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < u);

    #undef C
    #undef Cref
}

#endif        //  #ifndef CHECK_GEMM_HH
