#ifndef BLAS_HEMM_HH
#define BLAS_HEMM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"
#include "symm.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup hemm
inline
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc )
{
    symm( layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc );
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
inline
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc )
{
    symm( layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc );
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
inline
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc )
{
    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_msg_( lda < m, "lda %d < m %d", lda, m );
    else
        throw_if_msg_( lda < n, "lda %d < n %d", lda, n );

    if (layout == Layout::ColMajor) {
        throw_if_( ldb < m );
        throw_if_( ldc < m );
    }
    else {
        throw_if_( ldb < n );
        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap left <=> right, lower <=> upper, m <=> n
        side = (side == Side::Left  ? Side::Right : Side::Left);
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        std::swap( m_, n_ );
    }

    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    f77_chemm( &side_, &uplo_, &m_, &n_,
               &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
inline
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc )
{
    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( side != Side::Left &&
               side != Side::Right );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( m < 0 );
    throw_if_( n < 0 );

    if (side == Side::Left)
        throw_if_msg_( lda < m, "lda %d < m %d", lda, m );
    else
        throw_if_msg_( lda < n, "lda %d < n %d", lda, n );

    if (layout == Layout::ColMajor) {
        throw_if_( ldb < m );
        throw_if_( ldc < m );
    }
    else {
        throw_if_( ldb < n );
        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap left <=> right, lower <=> upper, m <=> n
        side = (side == Side::Left  ? Side::Right : Side::Left);
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        std::swap( m_, n_ );
    }

    char side_ = side2char( side );
    char uplo_ = uplo2char( uplo );
    f77_zhemm( &side_, &uplo_, &m_, &n_,
               &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// =============================================================================
/// Hermitian matrix-matrix multiply,
///     \f[ C = \alpha A B + \beta C, \f]
/// or
///     \f[ C = \alpha B A + \beta C, \f]
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
///     - Side::Left:  \f$ C = \alpha A B + \beta C, \f$
///     - Side::Right: \f$ C = \alpha B A + \beta C. \f$
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
///     - If side = Left:  lda >= max(1,m).
///     - If side = Right: lda >= max(1,n).
///
/// @param[in] B
///     The m-by-n matrix B,
///     stored in ldb-by-n array [RowMajor: m-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     ldb >= max(1,m) [RowMajor: ldb >= max(1,n)].
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
///     ldc >= max(1,m) [RowMajor: ldc >= max(1,n)].
///
/// @ingroup hemm

template< typename TA, typename TB, typename TC >
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    typename traits3<TA, TB, TC>::scalar_t alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    typename traits3<TA, TB, TC>::scalar_t beta,
    TC       *C, int64_t ldc )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMM_HH
