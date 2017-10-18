#ifndef BLAS_TRMM_HH
#define BLAS_TRMM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void trmm(
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( diag != Diag::NonUnit &&
               diag != Diag::Unit );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );

    if (layout == Layout::ColMajor)
        throw_if_( ldb < m );
    else
        throw_if_( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
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
    f77_strmm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
inline
void trmm(
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( diag != Diag::NonUnit &&
               diag != Diag::Unit );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );

    if (layout == Layout::ColMajor)
        throw_if_( ldb < m );
    else
        throw_if_( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
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
    f77_dtrmm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
inline
void trmm(
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( diag != Diag::NonUnit &&
               diag != Diag::Unit );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );

    if (layout == Layout::ColMajor)
        throw_if_( ldb < m );
    else
        throw_if_( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
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
    f77_ctrmm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// -----------------------------------------------------------------------------
inline
void trmm(
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( diag != Diag::NonUnit &&
               diag != Diag::Unit );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );

    if (layout == Layout::ColMajor)
        throw_if_( ldb < m );
    else
        throw_if_( ldb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
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
    f77_ztrmm( &side_, &uplo_, &trans_, &diag_, &m_, &n_, &alpha,
               A, &lda_, B, &ldb_ );
}

// =============================================================================
/// Triangular matrix-matrix multiply,
///     B = op(A)*B  or
///     B = B*op(A),
/// where op(A) is one of
///     op(A) = A    or
///     op(A) = A^T  or
///     op(A) = A^H,
/// B is an m-by-n matrix, and A is an m-by-m or n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///         Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] side
///         Whether op(A) is on the left or right of B:
///         side = Side::Left  is B = op(A)*B,
///         side = Side::Right is B = B*op(A).
///
/// @param[in] uplo
///         Whether A is upper or lower triangular.
///         uplo = Lower: A is lower triangular.
///         uplo = Upper: A is upper triangular.
///
/// @param[in] trans
///         The form of op(A):
///         trans = Op::NoTrans   is op(A) = A,
///         trans = Op::Trans     is op(A) = A^T,
///         trans = Op::ConjTrans is op(A) = A^H.
///
/// @param[in] diag
///         Whether A has a unit or non-unit diagonal:
///         diag = Diag::Unit    means A is assumed to be unit triangular,
///         diag = Diag::NonUnit means A is not assumed to be unit triangular.
///
/// @param[in] m
///         Number of rows of matrices B and B. m >= 0.
///
/// @param[in] n
///         Number of columns of matrices B and B. n >= 0.
///
/// @param[in] A
///         If side = Left,  the m-by-m matrix A, stored in an lda-by-m array.
///         If side = Right, the n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///         Leading dimension of A, i.e., column stride.
///         If side = left,  lda >= max(1, m).
///         If side = right, lda >= max(1, n).
///
/// @param[in,out] B
///         On entry, the m-by-n matrix B, stored in an ldb-by-n array.
///         On exit, overwritten by the solution matrix B.
///
/// @param[in] ldb
///         Leading dimension of B, i.e., column stride. ldb >= max(1, m).
///
/// @ingroup blas3

template< typename TA, typename TX >
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    typename blas::traits2<TA, TX>::scalar_t alpha,
    TA const *A, int64_t lda,
    TX       *B, int64_t ldb )
{
    throw std::exception();
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRMM_HH
