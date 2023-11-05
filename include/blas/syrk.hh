// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_SYRK_HH
#define BLAS_SYRK_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Symmetric rank-k update:
/// \[
///     C = \alpha A A^T + \beta C,
/// \]
/// or
/// \[
///     C = \alpha A^T A + \beta C,
/// \]
/// where alpha and beta are scalars, C is an n-by-n symmetric matrix,
/// and A is an n-by-k or k-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///     What part of the matrix C is referenced,
///     the opposite triangle being assumed from symmetry:
///     - Uplo::Lower: only the lower triangular part of C is referenced.
///     - Uplo::Upper: only the upper triangular part of C is referenced.
///
/// @param[in] trans
///     The operation to be performed:
///     - Op::NoTrans: $C = \alpha A A^T + \beta C$.
///     - Op::Trans:   $C = \alpha A^T A + \beta C$.
///     - In the real    case, Op::ConjTrans is interpreted as Op::Trans.
///       In the complex case, Op::ConjTrans is illegal (see @ref herk instead).
///
/// @param[in] n
///     Number of rows and columns of the matrix C. n >= 0.
///
/// @param[in] k
///     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
///     - Otherwise:          number of rows    of the matrix A. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not accessed.
///
/// @param[in] A
///     - If trans = NoTrans:
///       the n-by-k matrix A, stored in an lda-by-k array [RowMajor: n-by-lda].
///     - Otherwise:
///       the k-by-n matrix A, stored in an lda-by-n array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If trans = NoTrans: lda >= max(1, n) [RowMajor: lda >= max(1, k)],
///     - Otherwise:              lda >= max(1, k) [RowMajor: lda >= max(1, n)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The n-by-n symmetric matrix C,
///     stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, n).
///
/// @ingroup syrk

template <typename TA, typename TC>
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    scalar_type<TA, TC> alpha,
    TA const *A, int64_t lda,
    scalar_type<TA, TC> beta,
    TC       *C, int64_t ldc )
{
    typedef blas::scalar_type<TA, TC> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    #define C(i_, j_) C[ (i_) + (j_)*ldc ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper &&
                   uplo != Uplo::General );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    // check and interpret argument trans
    if (trans == Op::ConjTrans) {
        blas_error_if_msg(
                blas::is_complex<TA>::value,
                "trans == Op::ConjTrans && "
                "blas::is_complex<TA>::value" );
        trans = Op::Trans;
    }
    else {
        blas_error_if( trans != Op::NoTrans &&
                       trans != Op::Trans );
    }

    // adapt if row major
    if (layout == Layout::RowMajor) {
        if (uplo == Uplo::Lower)
            uplo = Uplo::Upper;
        else if (uplo == Uplo::Upper)
            uplo = Uplo::Lower;
        trans = (trans == Op::NoTrans)
                ? Op::Trans
                : Op::NoTrans;
    }

    // check remaining arguments
    blas_error_if( lda < ((trans == Op::NoTrans) ? n : k) );
    blas_error_if( ldc < n );

    // quick return
    if (n == 0 || k == 0)
        return;

    // alpha == zero
    if (alpha == zero) {
        if (beta == zero) {
            if (uplo != Uplo::Upper) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i <= j; ++i)
                        C(i, j) = zero;
                }
            }
            else if (uplo != Uplo::Lower) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = j; i < n; ++i)
                        C(i, j) = zero;
                }
            }
            else {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i < n; ++i)
                        C(i, j) = zero;
                }
            }
        }
        else if (beta != one) {
            if (uplo != Uplo::Upper) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i <= j; ++i)
                        C(i, j) *= beta;
                }
            }
            else if (uplo != Uplo::Lower) {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = j; i < n; ++i)
                        C(i, j) *= beta;
                }
            }
            else {
                for (int64_t j = 0; j < n; ++j) {
                    for (int64_t i = 0; i < n; ++i)
                        C(i, j) *= beta;
                }
            }
        }
        return;
    }

    // alpha != zero
    if (trans == Op::NoTrans) {
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for (int64_t j = 0; j < n; ++j) {

                for (int64_t i = 0; i <= j; ++i)
                    C(i, j) *= beta;

                for (int64_t l = 0; l < k; ++l) {
                    scalar_t alpha_Ajl = alpha*A(j, l);
                    for (int64_t i = 0; i <= j; ++i)
                        C(i, j) += A(i, l)*alpha_Ajl;
                }
            }
        }
        else { // uplo == Uplo::Lower
            for (int64_t j = 0; j < n; ++j) {

                for (int64_t i = j; i < n; ++i)
                    C(i, j) *= beta;

                for (int64_t l = 0; l < k; ++l) {
                    scalar_t alpha_Ajl = alpha*A(j, l);
                    for (int64_t i = j; i < n; ++i)
                        C(i, j) += A(i, l)*alpha_Ajl;
                }
            }
        }
    }
    else { // trans == Op::Trans
        if (uplo != Uplo::Lower) {
            // uplo == Uplo::Upper or uplo == Uplo::General
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i <= j; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l)
                        sum += A(l, i) * A(l, j);
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
        else { // uplo == Uplo::Lower
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = j; i < n; ++i) {
                    scalar_t sum = zero;
                    for (int64_t l = 0; l < k; ++l) {
                        sum +=  A(l, i) * A(l, j);
                    }
                    C(i, j) = alpha*sum + beta*C(i, j);
                }
            }
        }
    }

    if (uplo == Uplo::General) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = j+1; i < n; ++i)
                C(i, j) = C(j, i);
        }
    }

    #undef A
    #undef C
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMM_HH
