// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GEMM_HH
#define BLAS_GEMM_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// General matrix-matrix multiply:
/// \[
///     C = \alpha op(A) \times op(B) + \beta C,
/// \]
/// where $op(X)$ is one of
///     $op(X) = X$,
///     $op(X) = X^T$, or
///     $op(X) = X^H$,
/// alpha and beta are scalars, and A, B, and C are matrices, with
/// $op(A)$ an m-by-k matrix, $op(B)$ a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] transA
///     The operation $op(A)$ to be used:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] transB
///     The operation $op(B)$ to be used:
///     - Op::NoTrans:   $op(B) = B$.
///     - Op::Trans:     $op(B) = B^T$.
///     - Op::ConjTrans: $op(B) = B^H$.
///
/// @param[in] m
///     Number of rows of the matrix C and $op(A)$. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix C and $op(B)$. n >= 0.
///
/// @param[in] k
///     Number of columns of $op(A)$ and rows of $op(B)$. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
///     - Otherwise:
///       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
///     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
///
/// @param[in] B
///     - If transB = NoTrans:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///     - Otherwise:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an ldc-by-n array [RowMajor: m-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup gemm

template <typename TA, typename TB, typename TC>
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, int64_t ldc )
{
    // redirect if row major
    if (layout == Layout::RowMajor) {
        return gemm(
             Layout::ColMajor,
             transB,
             transA,
             n, m, k,
             alpha,
             B, ldb,
             A, lda,
             beta,
             C, ldc);
    }
    else {
        // check layout
        blas_error_if_msg( layout != Layout::ColMajor,
            "layout != Layout::ColMajor && layout != Layout::RowMajor" );
    }

    typedef blas::scalar_type<TA, TB, TC> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define B(i_, j_) B[ (i_) + (j_)*ldb ]
    #define C(i_, j_) C[ (i_) + (j_)*ldc ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    blas_error_if( lda < ((transA != Op::NoTrans) ? k : m) );
    blas_error_if( ldb < ((transB != Op::NoTrans) ? n : k) );
    blas_error_if( ldc < m );

    // quick return
    if (m == 0 || n == 0 || k == 0)
        return;

    // alpha == zero
    if (alpha == zero) {
        if (beta == zero) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i)
                    C(i, j) = zero;
            }
        }
        else if (beta != one) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
            }
        }
        return;
    }

    // alpha != zero
    if (transA == Op::NoTrans) {
        if (transB == Op::NoTrans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (int64_t l = 0; l < k; ++l) {
                    scalar_t alpha_Blj = alpha*B(l, j);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l)*alpha_Blj;
                }
            }
        }
        else if (transB == Op::Trans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (int64_t l = 0; l < k; ++l) {
                    scalar_t alpha_Bjl = alpha*B(j, l);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l)*alpha_Bjl;
                }
            }
        }
        else { // transB == Op::ConjTrans
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i)
                    C(i, j) *= beta;
                for (int64_t l = 0; l < k; ++l) {
                    scalar_t alpha_Bjl = alpha*conj(B(j, l));
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += A(i, l)*alpha_Bjl;
                }
            }
        }
    }
    else if (transA == Op::Trans) {
        if (transB == Op::NoTrans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += A(l, i)*B(l, j);
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += A(l, i)*B(j, l);
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += A(l, i)*conj(B(j, l));
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
    }
    else { // transA == Op::ConjTrans
        if (transB == Op::NoTrans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += conj(A(l, i))*B(l, j);
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
        else if (transB == Op::Trans) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += conj(A(l, i))*B(j, l);
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
        else { // transB == Op::ConjTrans
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += A(l, i)*B(j, l); // little improvement here
                    C(i, j) = alpha*conj(sum) + beta*C(i, j);
                }
            }
        }
    }

    #undef A
    #undef B
    #undef C
}

}  // namespace blas

#endif        //  #ifndef BLAS_GEMM_HH
