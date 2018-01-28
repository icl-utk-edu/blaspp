#ifndef BLAS_TRSM_HH
#define BLAS_TRSM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float       *B, int64_t ldb )
{
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

    if (side == Side::Left)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldb < m );
    else
        blas_error_if( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_strsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double       *B, int64_t ldb )
{
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

    if (side == Side::Left)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldb < m );
    else
        blas_error_if( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_dtrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *B, int64_t ldb )
{
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

    if (side == Side::Left)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldb < m );
    else
        blas_error_if( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_ctrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *B, int64_t ldb )
{
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

    if (side == Side::Left)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldb < m );
    else
        blas_error_if( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    char side_  = side2char( side );
    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    BLAS_ztrsm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// =============================================================================
/// Solve the triangular matrix-vector equation
///     \f[ op(A) X = \alpha B, \f]
/// or
///     \f[ X op(A) = \alpha B, \f]
/// where op(A) is one of
///     \f[ op(A) = A,   \f]
///     \f[ op(A) = A^T, \f]
///     \f[ op(A) = A^H, \f]
/// X and B are m-by-n matrices, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] side
///     Whether op(A) is on the left or right of X:
///     - Side::Left:  \f$ op(A) X = B. \f$
///     - Side::Right: \f$ X op(A) = B. \f$
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
///     Number of rows of matrices B and X. m >= 0.
///
/// @param[in] n
///     Number of columns of matrices B and X. n >= 0.
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
///     On entry,
///     the m-by-n matrix B, stored in an ldb-by-n array [RowMajor: m-by-ldb].
///     On exit, overwritten by the solution matrix X.
///
/// @param[in] ldb
///     Leading dimension of B. ldb >= max(1, m) [RowMajor: ldb >= max(1, n)].
///
/// @ingroup trsm

template< typename TA, typename TX >
void trsm(
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
    throw std::exception();
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRSM_HH
