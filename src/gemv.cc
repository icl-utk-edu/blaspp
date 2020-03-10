#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => A
        std::swap( m_, n_ );
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char trans_ = op2char( trans );
    BLAS_sgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => A
        std::swap( m_, n_ );
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char trans_ = op2char( trans );
    BLAS_dgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    // if x2=x, then it isn't modified
    std::complex<float> *x2 = const_cast< std::complex<float>* >( x );
    Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        // conjugate alpha, beta, x (in x2), and y (in-place)
        if (trans == Op::ConjTrans) {
            alpha = conj( alpha );
            beta  = conj( beta );

            x2 = new std::complex<float>[m];
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                x2[i] = conj( x[ix] );
                ix += incx;
            }
            incx_ = 1;

            int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t i = 0; i < n; ++i) {
                y[iy] = conj( y[iy] );
                iy += incy;
            }
        }
        // A => A^T; A^T => A; A^H => A + conj
        std::swap( m_, n_ );
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char trans_ = op2char( trans2 );
    BLAS_cgemv( &trans_, &m_, &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda_,
                (blas_complex_float*) x2, &incx_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) y, &incy_ );

    if (layout == Layout::RowMajor && trans == Op::ConjTrans) {
        // y = conj( y )
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = conj( y[iy] );
            iy += incy;
        }
        delete[] x2;
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( m              > std::numeric_limits<blas_int>::max() );
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int m_    = (blas_int) m;
    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    // if x2=x, then it isn't modified
    std::complex<double> *x2 = const_cast< std::complex<double>* >( x );
    Op trans2 = trans;
    if (layout == Layout::RowMajor) {
        // conjugate alpha, beta, x (in x2), and y (in-place)
        if (trans == Op::ConjTrans) {
            alpha = conj( alpha );
            beta  = conj( beta );

            x2 = new std::complex<double>[m];
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                x2[i] = conj( x[ix] );
                ix += incx;
            }
            incx_ = 1;

            int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t i = 0; i < n; ++i) {
                y[iy] = conj( y[iy] );
                iy += incy;
            }
        }
        // A => A^T; A^T => A; A^H => A + conj
        std::swap( m_, n_ );
        trans2 = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char trans_ = op2char( trans2 );
    BLAS_zgemv( &trans_, &m_, &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda_,
                (blas_complex_double*) x2, &incx_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) y, &incy_ );

    if (layout == Layout::RowMajor && trans == Op::ConjTrans) {
        // y = conj( y )
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = conj( y[iy] );
            iy += incy;
        }
        delete[] x2;
    }
}

}  // namespace blas
