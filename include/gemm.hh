#ifndef BLAS_GEMM_HH
#define BLAS_GEMM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( transA != Op::NoTrans &&
               transA != Op::Trans &&
               transA != Op::ConjTrans );
    throw_if_( transB != Op::NoTrans &&
               transB != Op::Trans &&
               transB != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    throw_if_( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB == Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB != Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( k   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
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
        f77_sgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        f77_sgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( transA != Op::NoTrans &&
               transA != Op::Trans &&
               transA != Op::ConjTrans );
    throw_if_( transB != Op::NoTrans &&
               transB != Op::Trans &&
               transB != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    throw_if_( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB == Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB != Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( k   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
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
        f77_dgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        f77_dgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( transA != Op::NoTrans &&
               transA != Op::Trans &&
               transA != Op::ConjTrans );
    throw_if_( transB != Op::NoTrans &&
               transB != Op::Trans &&
               transB != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    throw_if_( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB == Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB != Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( k   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
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
        f77_cgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        f77_cgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// -----------------------------------------------------------------------------
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( transA != Op::NoTrans &&
               transA != Op::Trans &&
               transA != Op::ConjTrans );
    throw_if_( transB != Op::NoTrans &&
               transB != Op::Trans &&
               transB != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    throw_if_( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB == Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            throw_if_( lda < m );
        else
            throw_if_( lda < k );

        if (transB != Op::NoTrans)
            throw_if_( ldb < k );
        else
            throw_if_( ldb < n );

        throw_if_( ldc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m   > std::numeric_limits<blas_int>::max() );
        throw_if_( n   > std::numeric_limits<blas_int>::max() );
        throw_if_( k   > std::numeric_limits<blas_int>::max() );
        throw_if_( lda > std::numeric_limits<blas_int>::max() );
        throw_if_( ldb > std::numeric_limits<blas_int>::max() );
        throw_if_( ldc > std::numeric_limits<blas_int>::max() );
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
        f77_zgemm( &transB_, &transA_, &n_, &m_, &k_,
                   &alpha, B, &ldb_, A, &lda_, &beta, C, &ldc_ );
    }
    else {
        f77_zgemm( &transA_, &transB_, &m_, &n_, &k_,
                   &alpha, A, &lda_, B, &ldb_, &beta, C, &ldc_ );
    }
}

// =============================================================================
/// General matrix-matrix multiply,
///     C = alpha*op(A)*op(B) + beta*C
/// where op(X) is one of
///     op(X) = X    or
///     op(X) = X^T  or
///     op(X) = X^H,
/// alpha and beta are scalars, and A, B, and C are matrices, with op(A)
/// an m-by-k matrix, op(B) a k-by-n matrix, and C an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///         Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] transA
///         The operation op(A) to be used:
///         transA = Op::NoTrans   is op(A) = A,
///         transA = Op::Trans     is op(A) = A^T,
///         transA = Op::ConjTrans is op(A) = A^H.
///
/// @param[in] transA
///         The operation op(B) to be used:
///         transB = Op::NoTrans   is op(B) = B,
///         transB = Op::Trans     is op(B) = B^T,
///         transB = Op::ConjTrans is op(B) = B^H.
///
/// @param[in] m
///         Number of rows of the matrix C and op(A).
///
/// @param[in] n
///         Number of columns of the matrix C and op(B).
///
/// @param[in] k
///         Number of columns of op(A) and rows of op(B).
///
/// @param[in] alpha
///         Scalar alpha. If alpha is zero, A and B are not accessed.
///
/// @param[in] A
///         NoTrans:     The m-by-k matrix A.
///         [Conj]Trans: The k-by-m matrix A.
///         ColMajor: stored in an lda-by-k or lda-by-m array.
///         RowMajor: stored in an m-by-lda or k-by-lda array.
///
/// @param[in] lda
///         Leading dimension of A.
///         NoTrans/ColMajor or [Conj]Trans/RowMajor: lda >= max(1,m).
///         [Conj]Trans/ColMajor or NoTrans/RowMajor: lda >= max(1,k).
///
/// @param[in] B
///         NoTrans:     The k-by-n matrix B.
///         [Conj]Trans: The n-by-k matrix B.
///         ColMajor: stored in an lda-by-n or lda-by-k array.
///         RowMajor: stored in an k-by-lda or n-by-lda array.
///
/// @param[in] ldb
///         Leading dimension of B, i.e., column stride.
///         NoTrans/ColMajor or [Conj]Trans/RowMajor: ldb >= max(1,n).
///         [Conj]Trans/ColMajor or NoTrans/RowMajor: ldb >= max(1,k).
///
/// @param[in] beta
///         Scalar beta. When beta is zero, C need not be set on input.
///
/// @param[in] C
///         The m-by-n matrix C, stored in an lda-by-n array.
///         ColMajor: lda >= max(1,n).
///         RowMajor: lda >= max(1,m).
///
/// @param[in] ldc
///         Leading dimension of C.
///         ldc >= max(1,m).
///
/// @ingroup blas3

template< typename TA, typename TB, typename TC >
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    typename traits3<TA, TB, TC>::scalar_t alpha,
    TA const *A, int64_t lda,
    TB const *B, int64_t ldb,
    typename traits3<TA, TB, TC>::scalar_t beta,
    TC       *C, int64_t ldc )
{
    typedef typename blas::traits3<TA, TB, TC>::scalar_t scalar_t;
}

}  // namespace blas

#endif        //  #ifndef BLAS_GEMM_HH
