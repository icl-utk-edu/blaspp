#include "blas_fortran.hh"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup gemm
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

}  // namespace blas
