// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_TRSV_HH
#define BLAS_TRSV_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Solve the triangular matrix-vector equation
/// \[
///     op(A) x = b,
/// \]
/// where $op(A)$ is one of
///     $op(A) = A$,
///     $op(A) = A^T$, or
///     $op(A) = A^H$,
/// x and b are vectors,
/// and A is an n-by-n, unit or non-unit, upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see LAPACK's latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero.
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The equation to be solved:
///     - Op::NoTrans:   $A   x = b$,
///     - Op::Trans:     $A^T x = b$,
///     - Op::ConjTrans: $A^H x = b$.
///
/// @param[in] diag
///     Whether A has a unit or non-unit diagonal:
///     - Diag::Unit:    A is assumed to be unit triangular.
///                      The diagonal elements of A are not referenced.
///     - Diag::NonUnit: A is not assumed to be unit triangular.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in, out] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @ingroup trsv

template< typename TA, typename TX >
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    TA const *A, int64_t lda,
    TX       *x, int64_t incx )
{
    typedef blas::scalar_type<TA, TX> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    // quick return
    if (n == 0)
        return;

    // for row major, swap lower <=> upper and
    // A => A^T; A^T => A; A^H => A & conj
    bool doconj = false;
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        if (trans == Op::NoTrans) {
            trans = Op::Trans;
        }
        else {
            if (trans == Op::ConjTrans) {
                doconj = true;
            }
            trans = Op::NoTrans;
        }
    }

    bool nonunit = (diag == Diag::NonUnit);
    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);

    if (trans == Op::NoTrans && ! doconj) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    if (nonunit) {
                        x[j] /= A(j, j);
                    }
                    TX tmp = x[j];
                    for (int64_t i = j - 1; i >= 0; --i) {
                        x[i] -= tmp * A(i, j);
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx + (n - 1)*incx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= A(j, j);
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        x[ix] -= tmp * A(i, j);
                    }
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[j] /= A(j, j);
                    }
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        x[i] -= tmp * A(i, j);
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= A(j, j);
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        x[ix] -= tmp * A(i, j);
                    }
                    jx += incx;
                }
            }
        }
    }
    else if (trans == Op::NoTrans && doconj) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    if (nonunit) {
                        x[j] /= conj( A(j, j) );
                    }
                    TX tmp = x[j];
                    for (int64_t i = j - 1; i >= 0; --i) {
                        x[i] -= tmp * conj( A(i, j) );
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx + (n - 1)*incx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= conj( A(j, j) );
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        x[ix] -= tmp * conj( A(i, j) );
                    }
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[j] /= conj( A(j, j) );
                    }
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        x[i] -= tmp * conj( A(i, j) );
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= conj( A(j, j) );
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        x[ix] -= tmp * conj( A(i, j) );
                    }
                    jx += incx;
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // Form  x := A^{-T} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= A(i, j) * x[i];
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    int64_t ix = kx;
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= A(i, j) * x[ix];
                        ix += incx;
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        tmp -= A(i, j) * x[i];
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                int64_t jx = kx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    int64_t ix = kx;
                    TX tmp = x[jx];
                    for (int64_t i = n - 1; i >= j + 1; --i) {
                        tmp -= A(i, j) * x[ix];
                        ix -= incx;
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
    }
    else {
        // Form x := A^{-H} * x
        // same code as above A^{-T} * x case, except add conj()
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= conj( A(i, j) ) * x[i];
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    int64_t ix = kx;
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= conj( A(i, j) ) * x[ix];
                        ix += incx;
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        tmp -= conj( A(i, j) ) * x[i];
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                int64_t jx = kx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    int64_t ix = kx;
                    TX tmp = x[jx];
                    for (int64_t i = n - 1; i >= j + 1; --i) {
                        tmp -= conj( A(i, j) ) * x[ix];
                        ix -= incx;
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRSV_HH
