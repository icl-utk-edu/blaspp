// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_HEMM_HH
#define BLAS_HEMM_HH

#include "blas/util.hh"
#include "blas/symm.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Hermitian matrix-matrix multiply:
/// \[
///     C = \alpha A B + \beta C,
/// \]
/// or
/// \[
///     C = \alpha B A + \beta C,
/// \]
/// where alpha and beta are scalars, A is an m-by-m or n-by-n Hermitian matrix,
/// and B and C are m-by-n matrices.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
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
///     - If side = Left:  the m-by-m matrix A,
///       stored in an lda-by-m array [RowMajor: m-by-lda].
///     - If side = Right: the n-by-n matrix A,
///       stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If side = Left:  lda >= max(1, m).
///     - If side = Right: lda >= max(1, n).
///
/// @param[in] B
///     The m-by-n matrix B,
///     stored in ldb-by-n array [RowMajor: m-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C,
///     stored in ldc-by-n array [RowMajor: m-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C.
///     ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup hemm

template< typename TA, typename TB, typename TC >
void hemm(
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
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMM_HH
