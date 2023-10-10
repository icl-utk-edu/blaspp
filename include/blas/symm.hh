// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYMM_HH
#define BLAS_SYMM_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Symmetric matrix-matrix multiply:
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C,
/// \]
/// where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric matrix,
/// and B and C are m-by-n matrices.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] side
///     The side the matrix A appears on:
///     - Side::Left:  $C = \alpha A B + \beta C$,
///     - Side::Right: $C = \alpha B A + \beta C$.
///
/// @param[in] uplo
///     What part of the matrix A is referenced:
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] m
///     Number of rows of the matrices B and C.
///
/// @param[in] n
///     Number of columns of the matrices B and C.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If side = Left:  The m-by-m matrix A, stored in an lda-by-m array.
///     - If side = Right: The n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = Left:  lda >= max(1, m).
///     - If side = Right: lda >= max(1, n).
///
/// @param[in] B
///     The m-by-n matrix B, stored in an ldb-by-n array.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, n).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an lda-by-n array.
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @ingroup symm

template <typename TA, typename TB, typename TC>
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, int64_t ldc )
{
    using std::swap;
    using scalar_t = blas::scalar_type<TA, TB>;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define B(i_, j_) B[ (i_) + (j_)*ldb ]
    #define C(i_, j_) C[ (i_) + (j_)*ldc ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
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
    blas_error_if( ldc < m );

    // quick return
    if (m == 0 || n == 0)
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
    if (side == Side::Left) {
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i < m; ++i) {

                    scalar_t alpha_Bij = alpha*B(i, j);
                    scalar_t sum = zero;

                    for (int64_t k = 0; k < i; ++k) {
                        C(k, j) += A(k, i) * alpha_Bij;
                        sum += A(k, i) * B(k, j);
                    }
                    C(i, j) =
                        beta * C(i, j)
                        + A(i, i) * alpha_Bij
                        + alpha * sum;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = m-1; i >= 0; --i) {

                    scalar_t alpha_Bij = alpha*B(i, j);
                    scalar_t sum = zero;

                    for (int64_t k = i+1; k < m; ++k) {
                        C(k, j) += A(k, i) * alpha_Bij;
                        sum += A(k, i) * B(k, j);
                    }
                    C(i, j) =
                        beta * C(i, j)
                        + A(i, i) * alpha_Bij
                        + alpha * sum;
                }
            }
        }
    }
    else { // side == Side::Right
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for (int64_t j = 0; j < n; ++j) {

                scalar_t alpha_Akj = alpha * A(j, j);

                for (int64_t i = 0; i < m; ++i)
                    C(i, j) = beta * C(i, j) + B(i, j) * alpha_Akj;

                for (int64_t k = 0; k < j; ++k) {
                    alpha_Akj = alpha*A(k, j);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += B(i, k) * alpha_Akj;
                }

                for (int64_t k = j+1; k < n; ++k) {
                    alpha_Akj = alpha * A(j, k);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += B(i, k) * alpha_Akj;
                }
            }
        }
        else {
            // uplo == Uplo::Lower
            for (int64_t j = 0; j < n; ++j) {

                scalar_t alpha_Akj = alpha * A(j, j);

                for (int64_t i = 0; i < m; ++i)
                    C(i, j) = beta * C(i, j) + B(i, j) * alpha_Akj;

                for (int64_t k = 0; k < j; ++k) {
                    alpha_Akj = alpha * A(j, k);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += B(i, k) * alpha_Akj;
                }

                for (int64_t k = j+1; k < n; ++k) {
                    alpha_Akj = alpha*A(k, j);
                    for (int64_t i = 0; i < m; ++i)
                        C(i, j) += B(i, k) * alpha_Akj;
                }
            }
        }
    }

    #undef A
    #undef B
    #undef C
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMM_HH
