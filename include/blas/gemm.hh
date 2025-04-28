// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_GEMM_HH
#define BLAS_GEMM_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

//------------------------------------------------------------------------------
/// @return ceil( x / y ), for integer types T1, T2.
template <typename T1, typename T2>
inline constexpr std::common_type_t<T1, T2> ceildiv( T1 x, T2 y )
{
    using T = std::common_type_t<T1, T2>;
    return T((x + y - 1) / y);
}

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
    if (m == 0 || n == 0 || ((alpha == zero || k == 0) && beta == one))
        return;

    // Handle rare alpha == zero case as degenerate k.
    if (alpha == zero)
        k = 0;

    // Simple, single-threaded, blocked gemm algorithm.
    // Block sizes can be tuned for specific datatypes and architectures.
    const int mb = 32, nb = 16, kb = 8;

    // Macros to transpose indices to C row-major ordering.
    #define sA( i_, j_ ) sA[ j_ ][ i_ ]
    #define sB( i_, j_ ) sB[ j_ ][ i_ ]
    #define sC( i_, j_ ) sC[ j_ ][ i_ ]

    scalar_t sA[ kb ][ mb ];
    scalar_t sB[ nb ][ kb ];

    for (int64_t i = 0; i < m; i += mb) {
        int mb_ = min( mb, m - i );
        for (int64_t j = 0; j < n; j += nb) {
            int nb_ = min( nb, n - j );

            // Zero tile of C.
            scalar_t sC[ nb ][ mb ] = { 0 };

            for (int h = 0; h < k; h += kb) {
                int kb_ = min( kb, k - h );

                // Load tile of A, applying trans/conj as needed.
                // If inner loop bound is known, use that to enable loop
                // unrolling and vectorizing.
                //printf( "load A( %d, %d )\n", i, h );
                if (transA == Op::NoTrans) {
                    int64_t ih_offset = i + h*lda;
                    if (mb == mb_) {
                        for (int hh = 0; hh < kb_; ++hh) {
                            for (int ii = 0; ii < mb; ++ii) {  // fixed
                                sA( ii, hh ) = A( ih_offset + ii, hh );
                            }
                        }
                    }
                    else {
                    //printf( "A: lim mb %d, %d\n", mb, mb_ );
                        for (int hh = 0; hh < kb_; ++hh) {
                            for (int ii = 0; ii < mb_; ++ii) {
                                sA( ii, hh ) = A( ih_offset + ii, hh );
                            }
                        }
                    }
                    // Clear k-edge entries.
                    for (int hh = kb_; hh < kb; ++hh) {
                        for (int ii = 0; ii < mb; ++ii) {  // fixed
                            sA( ii, hh ) = 0;
                        }
                    }
                }
                else if (transA == Op::Trans) {
                    int64_t ih_offset = i*lda + h;
                    if (kb == kb_) {
                        for (int ii = 0; ii < mb_; ++ii) {
                            for (int hh = 0; hh < kb; ++hh) {  // fixed
                                sA( ii, hh ) = A( ih_offset + hh, ii );
                            }
                        }
                    }
                    else {
                    // printf( "A: lim kb %d, %d\n", kb, kb_ );
                        for (int ii = 0; ii < mb_; ++ii) {
                            for (int hh = 0; hh < kb_; ++hh) {
                                sA( ii, hh ) = A( ih_offset + hh, ii );
                            }
                            // Clear k-edge entries.
                            for (int hh = kb_; hh < kb; ++hh) {
                                sA( ii, hh ) = 0;
                            }
                        }
                    }
                }
                else if (transA == Op::ConjTrans) {
                    int64_t ih_offset = i*lda + h;
                    if (kb == kb_) {
                        for (int ii = 0; ii < mb_; ++ii) {
                            for (int hh = 0; hh < kb; ++hh) {  // fixed
                                sA( ii, hh ) = conj( A( ih_offset + hh, ii ) );
                            }
                        }
                    }
                    else {
                        for (int ii = 0; ii < mb_; ++ii) {
                            for (int hh = 0; hh < kb_; ++hh) {
                                sA( ii, hh ) = conj( A( ih_offset + hh, ii ) );
                            }
                            // Clear k-edge entries.
                            for (int hh = kb_; hh < kb; ++hh) {
                                sA( ii, hh ) = 0;
                            }
                        }
                    }
                }

                // Load tile of B, applying trans/conj as needed.
                //printf( "load B( %d, %d )\n", h, j );
                if (transB == Op::NoTrans) {
                    int64_t hj_offset = h + j*ldb;
                    if (kb == kb_) {
                        for (int jj = 0; jj < nb_; ++jj) {
                            for (int hh = 0; hh < kb; ++hh) {  // fixed
                                sB( hh, jj ) = B( hj_offset + hh, jj );
                            }
                        }
                    }
                    else {
                    // printf( "B: lim kb %d, %d\n", kb, kb_ );
                        for (int jj = 0; jj < nb_; ++jj) {
                            for (int hh = 0; hh < kb_; ++hh) {
                                sB( hh, jj ) = B( hj_offset + hh, jj );
                            }
                            // Clear k-edge entries.
                            for (int hh = kb_; hh < kb; ++hh) {
                                sB( hh, jj ) = 0;
                            }
                        }
                    }
                }
                else {
                    int64_t hj_offset = h*ldb + j;
                    if (transB == Op::Trans) {
                        if (nb == nb_) {
                            for (int hh = 0; hh < kb_; ++hh) {
                                for (int jj = 0; jj < nb; ++jj) {  // fixed
                                    sB( hh, jj ) = B( hj_offset + jj, hh );
                                }
                            }
                        }
                        else {
                        // printf( "B: lim nb %d, %d\n", nb, nb_ );
                            for (int hh = 0; hh < kb_; ++hh) {
                                for (int jj = 0; jj < nb_; ++jj) {
                                    sB( hh, jj ) = B( hj_offset + jj, hh );
                                }
                            }
                        }
                    }
                    else {  // transB == Op::ConjTrans
                        if (nb == nb_) {
                            for (int hh = 0; hh < kb_; ++hh) {
                                for (int jj = 0; jj < nb; ++jj) {  // fixed
                                    sB( hh, jj ) = conj( B( hj_offset + jj, hh ) );
                                }
                            }
                        }
                        else {
                            for (int hh = 0; hh < kb_; ++hh) {
                                for (int jj = 0; jj < nb_; ++jj) {
                                    sB( hh, jj ) = conj( B( hj_offset + jj, hh ) );
                                }
                            }
                        }
                    }
                    // Clear k-edge entries.
                    for (int hh = kb_; hh < kb; ++hh) {
                        for (int jj = 0; jj < nb; ++jj) {  // fixed
                            sB( hh, jj ) = 0;
                        }
                    }
                }

                // Multiply tiles: sC = sA * sB.
                //printf( "multiply A*B\n" );
                for (int jj = 0; jj < nb; ++jj) {
                    for (int ii = 0; ii < mb; ++ii) {
                        scalar_t sum = 0;
                        for (int hh = 0; hh < kb; ++hh) {
                            sum += sA( ii, hh ) * sB( hh, jj );
                            //sum += A( ih_offset + ii, hh ) * B( hj_offset + hh, jj );
                        }
                        sC( ii, jj ) += sum;
                    }
                }
            }

            // Apply alpha and beta, then store tile of C.
            //printf( "save C( %d, %d )\n", i, j  );
            int64_t ij_offset = i + j*ldc;
            if (beta == zero) {
                for (int jj = 0; jj < nb_; ++jj) {
                    for (int ii = 0; ii < mb_; ++ii) {
                        C( ij_offset + ii, jj ) = alpha*sC( ii, jj );
                    }
                }
            }
            else {
                for (int jj = 0; jj < nb_; ++jj) {
                    for (int ii = 0; ii < mb_; ++ii) {
                        C( ij_offset + ii, jj ) = alpha*sC( ii, jj )
                                                + beta*C( ij_offset + ii, jj );
                    }
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
