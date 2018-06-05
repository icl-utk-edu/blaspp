#include "blas_fortran.hh"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup symm
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

}  // namespace blas
