#ifndef BLAS_GEMM_HH
#define BLAS_GEMM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((transA == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lda < m );
    else
        blas_error_if( lda < k );

    if ((transB == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldb < k );
    else
        blas_error_if( ldb < n );

    if (layout == Layout::ColMajor)
        blas_error_if( ldc < m );
    else
        blas_error_if( ldc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    char transA_ = op2char( transA );
    char transB_ = op2char( transB );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        BLAS_sgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        BLAS_sgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB == Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB != Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    char transA_ = op2char( transA );
    char transB_ = op2char( transB );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        BLAS_dgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        BLAS_dgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB == Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB != Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    char transA_ = op2char( transA );
    char transB_ = op2char( transB );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        BLAS_cgemm( &transB_, &transA_, &n_, &m_, &k_,
                    (blas_complex_float*) &alpha,
                    (blas_complex_float*) B, &ldb_,
                    (blas_complex_float*) A, &lda_,
                    (blas_complex_float*) &beta,
                    (blas_complex_float*) C, &ldc_ );
    }
    else {
        BLAS_cgemm( &transA_, &transB_, &m_, &n_, &k_,
                    (blas_complex_float*) &alpha,
                    (blas_complex_float*) A, &lda_,
                    (blas_complex_float*) B, &ldb_,
                    (blas_complex_float*) &beta,
                    (blas_complex_float*) C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB == Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB != Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m   > std::numeric_limits<blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldb > std::numeric_limits<blas_int>::max() );
        blas_error_if( ldc > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_   = (blas_int) m;
    blas_int n_   = (blas_int) n;
    blas_int k_   = (blas_int) k;
    blas_int lda_ = (blas_int) lda;
    blas_int ldb_ = (blas_int) ldb;
    blas_int ldc_ = (blas_int) ldc;

    char transA_ = op2char( transA );
    char transB_ = op2char( transB );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        BLAS_zgemm( &transB_, &transA_, &n_, &m_, &k_,
                    (blas_complex_double*) &alpha,
                    (blas_complex_double*) B, &ldb_,
                    (blas_complex_double*) A, &lda_,
                    (blas_complex_double*) &beta,
                    (blas_complex_double*) C, &ldc_ );
    }
    else {
        BLAS_zgemm( &transA_, &transB_, &m_, &n_, &k_,
                    (blas_complex_double*) &alpha,
                    (blas_complex_double*) A, &lda_,
                    (blas_complex_double*) B, &ldb_,
                    (blas_complex_double*) &beta,
                    (blas_complex_double*) C, &ldc_ );
    }
}

// =============================================================================
/// General matrix-matrix multiply,
///     \f[ C = \alpha op(A) \times op(B) + \beta C, \f]
/// where op(X) is one of
///     \f[ op(X) = X,   \f]
///     \f[ op(X) = X^T, \f]
///     \f[ op(X) = X^H, \f]
/// alpha and beta are scalars, and A, B, and C are matrices, with
/// op(A) an m-by-k matrix, op(B) a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] transA
///     The operation op(A) to be used:
///     - Op::NoTrans:   \f$ op(A) = A.   \f$
///     - Op::Trans:     \f$ op(A) = A^T. \f$
///     - Op::ConjTrans: \f$ op(A) = A^H. \f$
///
/// @param[in] transB
///     The operation op(B) to be used:
///     - Op::NoTrans:   \f$ op(B) = B.   \f$
///     - Op::Trans:     \f$ op(B) = B^T. \f$
///     - Op::ConjTrans: \f$ op(B) = B^H. \f$
///
/// @param[in] m
///     Number of rows of the matrix C and op(A). m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix C and op(B). n >= 0.
///
/// @param[in] k
///     Number of columns of op(A) and rows of op(B). k >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///     - If transA = NoTrans:
///       the m-by-k matrix A, stored in an lda-by-k array [RowMajor: m-by-lda].
///     - Otherwise:
///       the k-by-m matrix A, stored in an lda-by-m array [RowMajor: k-by-lda].
///
/// @param[in] lda
///     Leading dimension of A.
///     - If transA = NoTrans: lda >= max(1, m) [RowMajor: lda >= max(1, k)].
///     - Otherwise:           lda >= max(1, k) [RowMajor: lda >= max(1, m)].
///
/// @param[in] B
///     - If transB = NoTrans:
///       the k-by-n matrix B, stored in an ldb-by-n array [RowMajor: k-by-ldb].
///     - Otherwise:
///       the n-by-k matrix B, stored in an ldb-by-k array [RowMajor: n-by-ldb].
///
/// @param[in] ldb
///     Leading dimension of B.
///     - If transB = NoTrans: ldb >= max(1, k) [RowMajor: ldb >= max(1, n)].
///     - Otherwise:           ldb >= max(1, n) [RowMajor: ldb >= max(1, k)].
///
/// @param[in] beta
///     Scalar beta. If beta is zero, C need not be set on input.
///
/// @param[in] C
///     The m-by-n matrix C, stored in an ldc-by-n array [RowMajor: m-by-ldc].
///
/// @param[in] ldc
///     Leading dimension of C. ldc >= max(1, m) [RowMajor: ldc >= max(1, n)].
///
/// @ingroup gemm

template< typename TA, typename TB, typename TC >
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    scalar_type<TA, TB, TC> alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    scalar_type<TA, TB, TC> beta,
    TC       *C, int64_t ldc )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_GEMM_HH
