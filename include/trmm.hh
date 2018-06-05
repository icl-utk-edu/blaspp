#ifndef BLAS_TRMM_HH
#define BLAS_TRMM_HH

#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Triangular matrix-matrix multiply,
///     \f[ B = \alpha op(A) B, \f]
/// or
///     \f[ B = \alpha B op(A), \f]
/// where op(A) is one of
///     \f[ op(A) = A,   \f]
///     \f[ op(A) = A^T, \f]
///     \f[ op(A) = A^H, \f]
/// B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] side
///     Whether op(A) is on the left or right of B:
///     - Side::Left:  \f$ B = \alpha op(A) B. \f$
///     - Side::Right: \f$ B = \alpha B op(A). \f$
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed to be zero:
///     - Uplo::Lower: A is lower triangular.
///     - Uplo::Upper: A is upper triangular.
///
/// @param[in] trans
///     The form of op(A):
///     - Op::NoTrans:   \f$ op(A) = A.   \f$
///     - Op::Trans:     \f$ op(A) = A^T. \f$
///     - Op::ConjTrans: \f$ op(A) = A^H. \f$
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

template< typename TA, typename TX >
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    blas::scalar_type<TA, TX> alpha,
    TA const *A, int64_t lda,
    TX       *B, int64_t ldb )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMM_HH
