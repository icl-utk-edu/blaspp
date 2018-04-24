#ifndef BLAS_SYMM_HH
#define BLAS_SYMM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup symm
inline
void symm(
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
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( lda < m, "lda %lld < m %lld", (lld) lda, (lld) m );
    else
        blas_error_if_msg( lda < n, "lda %lld < n %lld", (lld) lda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( ldb < m );
        blas_error_if( ldc < m );
    }
    else {
        blas_error_if( ldb < n );
        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
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
    BLAS_ssymm( &side_, &uplo_, &m_, &n_,
               &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symm
inline
void symm(
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
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( lda < m, "lda %lld < m %lld", (lld) lda, (lld) m );
    else
        blas_error_if_msg( lda < n, "lda %lld < n %lld", (lld) lda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( ldb < m );
        blas_error_if( ldc < m );
    }
    else {
        blas_error_if( ldb < n );
        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
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
    BLAS_dsymm( &side_, &uplo_, &m_, &n_,
               &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symm
inline
void symm(
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
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( lda < m, "lda %lld < m %lld", (lld) lda, (lld) m );
    else
        blas_error_if_msg( lda < n, "lda %lld < n %lld", (lld) lda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( ldb < m );
        blas_error_if( ldc < m );
    }
    else {
        blas_error_if( ldb < n );
        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
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
    BLAS_csymm( &side_, &uplo_, &m_, &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda_,
                (blas_complex_float*) B, &ldb_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symm
inline
void symm(
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
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( lda < m, "lda %lld < m %lld", (lld) lda, (lld) m );
    else
        blas_error_if_msg( lda < n, "lda %lld < n %lld", (lld) lda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( ldb < m );
        blas_error_if( ldc < m );
    }
    else {
        blas_error_if( ldb < n );
        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
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
    BLAS_zsymm( &side_, &uplo_, &m_, &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda_,
                (blas_complex_double*) B, &ldb_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &ldc_ );
}

// =============================================================================
/// Symmetric matrix-matrix multiply,
///     \f[ C = \alpha A B + \beta C, \f]
/// or
///     \f[ C = \alpha B A + \beta C, \f]
/// where alpha and beta are scalars, A is an m-by-m or n-by-n symmetric matrix,
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

template< typename TA, typename TB, typename TC >
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
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMM_HH
