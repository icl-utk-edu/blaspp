#ifndef BLAS_HER2K_HH
#define BLAS_HER2K_HH

#include "blas_fortran.hh"
#include "blas_util.hh"
#include "syr2k.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup her2k
inline
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc )
{
    syr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

// -----------------------------------------------------------------------------
/// @ingroup her2k
inline
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc )
{
    syr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

// -----------------------------------------------------------------------------
/// @ingroup her2k
inline
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,  // note: complex
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    float beta,   // note: real
    std::complex<float>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor)) {
        blas_error_if( lda < n );
        blas_error_if( ldb < n );
    }
    else {
        blas_error_if( lda < k );
        blas_error_if( ldb < k );
    }

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_cher2k( &uplo_, &trans_, &n_, &k_,
                &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup her2k
inline
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,  // note: complex
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    double beta,  // note: real
    std::complex<double>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor)) {
        blas_error_if( lda < n );
        blas_error_if( ldb < n );
    }
    else {
        blas_error_if( lda < k );
        blas_error_if( ldb < k );
    }

    blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::ConjTrans : Op::NoTrans);
    }

    char uplo_ = uplo2char( uplo );
    char trans_ = op2char( trans );
    BLAS_zher2k( &uplo_, &trans_, &n_, &k_,
                &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
}

// =============================================================================
/// Hermitian rank-k update,
///     \f[ C = \alpha A B^H + conj(\alpha) A^H B + \beta C, \f]
/// or
///     \f[ C = \alpha A^H B + conj(\alpha) B A^H + \beta C, \f]
/// where alpha and beta are scalars, C is an n-by-n Hermitian matrix,
/// and A and B are n-by-k or k-by-n matrices.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
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
///     - Op::NoTrans:   \f$ C = \alpha A B^H + conj(\alpha) A^H B + \beta C. \f$
///     - Op::ConjTrans: \f$ C = \alpha A^H B + conj(\alpha) B A^H + \beta C. \f$
///     - In the real    case, Op::Trans is interpreted as Op::ConjTrans.
///       In the complex case, Op::Trans is illegal (see @ref syr2k instead).
///
/// @param[in] n
///     Number of rows and columns of the matrix C. n >= 0.
///
/// @param[in] k
///     - If trans = NoTrans: number of columns of the matrix A. k >= 0.
///     - Otherwise:          number of rows    of the matrix A. k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If trans = NoTrans:
///       the n-by-k matrix A, stored in an lda-by-k array [RowMajor: n-by-lda].
///     - Otherwise:
///       the k-by-n matrix A, stored in an lda-by-n array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If trans = NoTrans: lda >= max(1,n) [RowMajor: lda >= max(1,k)],
///     - Otherwise:          lda >= max(1,k) [RowMajor: lda >= max(1,n)].
///
/// @param[in] B
///     - If trans = NoTrans:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///     - Otherwise:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If trans = NoTrans: ldb >= max(1,n) [RowMajor: ldb >= max(1,k)],
///     - Otherwise:          ldb >= max(1,k) [RowMajor: ldb >= max(1,n)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The n-by-n Hermitian matrix C,
///     stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1,n).
///
/// @ingroup her2k

template< typename TA, typename TB, typename TC >
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    typename traits3<TA, TB, TC>::scalar_t alpha,  // note: complex
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    typename traits3<TA, TB, TC>::real_t beta,  // note: real
    TC       *C, int64_t ldc )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER2K_HH
