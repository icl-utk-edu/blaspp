// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRMM_HH
#define BLAS_TRMM_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Triangular matrix-matrix multiply:
/// \[
///     B = \alpha op(A) B,
/// \]
/// or
/// \[
///     B = \alpha B op(A),
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] side
///     Whether $op(A)$ is on the left or right of B:
///     - Side::Left:  $B = \alpha op(A) B$.
///     - Side::Right: $B = \alpha B op(A)$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///     - Uplo::General is illegal (see @ref gemm instead).
///
/// @param[in] trans
///     The form of $op(A)$:
///     - Op::NoTrans:   $op(A) = A$.
///     - Op::Trans:     $op(A) = A^T$.
///     - Op::ConjTrans: $op(A) = A^H$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] m
///     Number of rows of matrix B. m >= 0.
///
/// @param[in] n
///     Number of columns of matrix B. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not accessed.
///
/// @param[in] A
///     - If side = Left:
///       the m-by-m matrix A, stored in an lda-by-m array [RowMajor: m-by-lda].
///     - If side = Right:
///       the n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = left:  lda >= max(1, m).
///     - If side = right: lda >= max(1, n).
///
/// @param[in, out] B
///     The m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @ingroup trmm

template <typename TA, typename TB>
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    blas::scalar_type<TA, TB> alpha,
    TA const *A, int64_t lda,
    TB       *B, int64_t ldb )
{
    using std::swap;
    using scalar_t = blas::scalar_type<TA, TB>;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define B(i_, j_) B[ (i_) + (j_)*ldb ]

    // constants
    const scalar_t zero = 0;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    // adapt if row major
    if (layout == Layout::RowMajor) {
        side = (side == Side::Left)
               ? Side::Right
               : Side::Left;
        if (uplo == Uplo::Lower)
            uplo = Uplo::Upper;
        else if (uplo == Uplo::Upper)
            uplo = Uplo::Lower;
        swap( m, n );
    }

    // check remaining arguments
    blas_error_if( lda < ((side == Side::Left) ? m : n) );
    blas_error_if( ldb < m );

    // quick return
    if (m == 0 || n == 0)
        return;

    // alpha == zero
    if (alpha == zero) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < m; ++i)
                B(i, j) = zero;
        }
        return;
    }

    // alpha != zero
    if (side == Side::Left) {
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t k = 0; k < m; ++k) {
                        scalar_t alpha_Bkj = alpha*B(k, j);
                        for (int64_t i = 0; i < k; ++i)
                            B(i, j) += A(i, k)*alpha_Bkj;
                        B(k, j) = (diag == Diag::NonUnit)
                                  ? A(k, k)*alpha_Bkj
                                  : alpha_Bkj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t k = m-1; k >= 0; --k) {
                        scalar_t alpha_Bkj = alpha*B(k, j);
                        B(k, j) = (diag == Diag::NonUnit)
                                  ? A(k, k)*alpha_Bkj
                                  : alpha_Bkj;
                        for (int64_t i = k+1; i < m; ++i)
                            B(i, j) += A(i, k)*alpha_Bkj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = m-1; i >= 0; --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                       ? A(i, i)*B(i, j)
                                       : B(i, j);
                        for (int64_t k = 0; k < i; ++k)
                            sum += A(k, i)*B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                       ? A(i, i)*B(i, j)
                                       : B(i, j);
                        for (int64_t k = i+1; k < m; ++k)
                            sum += A(k, i)*B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = m-1; i >= 0; --i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                       ? conj(A(i, i))*B(i, j)
                                       : B(i, j);
                        for (int64_t k = 0; k < i; ++k)
                            sum += conj(A(k, i))*B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i < m; ++i) {
                        scalar_t sum = (diag == Diag::NonUnit)
                                       ? conj(A(i, i))*B(i, j)
                                       : B(i, j);
                        for (int64_t k = i+1; k < m; ++k)
                            sum += conj(A(k, i))*B(k, j);
                        B(i, j) = alpha * sum;
                    }
                }
            }
        }
    }
    else { // side == Side::Right
        if (trans == Op::NoTrans) {
            if (uplo == Uplo::Upper) {
                for (int64_t j = n-1; j >= 0; --j) {

                    scalar_t alpha_Akj = (diag == Diag::NonUnit)
                                         ? alpha*A(j, j)
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, j) *= alpha_Akj;

                    for (int64_t k = 0; k < j; ++k) {
                        alpha_Akj = alpha*A(k, j);
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Akj;
                    }
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t j = 0; j < n; ++j) {

                    scalar_t alpha_Akj = (diag == Diag::NonUnit)
                                         ? alpha*A(j, j)
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, j) *= alpha_Akj;

                    for (int64_t k = j+1; k < n; ++k) {
                        alpha_Akj = alpha*A(k, j);
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Akj;
                    }
                }
            }
        }
        else if (trans == Op::Trans) {
            if (uplo == Uplo::Upper) {
                for (int64_t k = 0; k < n; ++k) {
                    for (int64_t j = 0; j < k; ++j) {
                        scalar_t alpha_Ajk = alpha*A(j, k);
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Ajk;
                    }

                    scalar_t alpha_Akk = (diag == Diag::NonUnit)
                                         ? alpha*A(k, k)
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, k) *= alpha_Akk;
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t k = n-1; k >= 0; --k) {
                    for (int64_t j = k+1; j < n; ++j) {
                        scalar_t alpha_Ajk = alpha*A(j, k);
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Ajk;
                    }

                    scalar_t alpha_Akk = (diag == Diag::NonUnit)
                                         ? alpha*A(k, k)
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, k) *= alpha_Akk;
                }
            }
        }
        else { // trans == Op::ConjTrans
            if (uplo == Uplo::Upper) {
                for (int64_t k = 0; k < n; ++k) {
                    for (int64_t j = 0; j < k; ++j) {
                        scalar_t alpha_Ajk = alpha*conj(A(j, k));
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Ajk;
                    }

                    scalar_t alpha_Akk = (diag == Diag::NonUnit)
                                         ? alpha*conj(A(k, k))
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, k) *= alpha_Akk;
                }
            }
            else { // uplo == Uplo::Lower
                for (int64_t k = n-1; k >= 0; --k) {
                    for (int64_t j = k+1; j < n; ++j) {
                        scalar_t alpha_Ajk = alpha*conj(A(j, k));
                        for (int64_t i = 0; i < m; ++i)
                            B(i, j) += B(i, k)*alpha_Ajk;
                    }

                    scalar_t alpha_Akk = (diag == Diag::NonUnit)
                                         ? alpha*conj(A(k, k))
                                         : alpha;
                    for (int64_t i = 0; i < m; ++i)
                        B(i, k) *= alpha_Akk;
                }
            }
        }
    }

    #undef A
    #undef B
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMM_HH
