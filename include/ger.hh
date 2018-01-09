#ifndef BLAS_GER_HH
#define BLAS_GER_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup ger
inline
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

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
        // swap m <=> n, x <=> y
        BLAS_sger( &n_, &m_, &alpha, y, &incy_, x, &incx_, A, &lda_ );
    }
    else {
        BLAS_sger( &m_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
inline
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

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
        // swap m <=> n, x <=> y
        BLAS_dger( &n_, &m_, &alpha, y, &incy_, x, &incx_, A, &lda_ );
    }
    else {
        BLAS_dger( &m_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
inline
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

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
        // conjugate y (in y2)
        std::complex<float> *y2 = new std::complex<float>[n];
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y2[i] = conj( y[iy] );
            iy += incy;
        }
        incy_ = 1;

        // swap m <=> n, x <=> y, call geru
        BLAS_cgeru( &n_, &m_,
                   &alpha, y2, &incy_, x, &incx_, A, &lda_ );

        delete[] y2;
    }
    else {
        BLAS_cgerc( &m_, &n_,
                   &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// -----------------------------------------------------------------------------
/// @ingroup ger
inline
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

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
        // conjugate y (in y2)
        std::complex<double> *y2 = new std::complex<double>[n];
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y2[i] = conj( y[iy] );
            iy += incy;
        }
        incy_ = 1;

        // swap m <=> n, x <=> y, call geru
        BLAS_zgeru( &n_, &m_,
                   &alpha, y2, &incy_, x, &incx_, A, &lda_ );

        delete[] y2;
    }
    else {
        BLAS_zgerc( &m_, &n_,
                   &alpha, x, &incx_, y, &incy_, A, &lda_ );
    }
}

// =============================================================================
/// General matrix rank-1 update,
///     \f[ A = \alpha x y^H + A, \f]
/// where alpha is a scalar, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] m
///     Number of rows of the matrix A. m >= 0.
///
/// @param[in] n
///     Number of columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The m-element vector x, in an array of length (m-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param[in,out] A
///     The m-by-n matrix A, stored in an lda-by-n array [RowMajor: m-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1,m) [RowMajor: lda >= max(1,n)].
///
/// @ingroup ger

template< typename TA, typename TX, typename TY >
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    typename blas::traits3<TA, TX, TY>::scalar_t alpha,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy,
    TA *A, int64_t lda )
{
    typedef typename blas::traits3<TA, TX, TY>::scalar_t scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (layout == Layout::ColMajor)
        blas_error_if( lda < m );
    else
        blas_error_if( lda < n );

    // quick return
    if (m == 0 || n == 0 || alpha == zero)
        return;

    if (layout == Layout::ColMajor) {
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if y[j] is zero, for consistent NAN handling
                scalar_t tmp = alpha * conj( y[j] );
                for (int64_t i = 0; i < m; ++i) {
                    A(i,j) += x[i] * tmp;
                }
            }
        }
        else if (incx == 1) {
            // x unit stride, y non-unit stride
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( y[jy] );
                for (int64_t i = 0; i < m; ++i) {
                    A(i,j) += x[i] * tmp;
                }
                jy += incy;
            }
        }
        else {
            // x and y non-unit stride
            int64_t kx = (incx > 0 ? 0 : (-m + 1)*incx);
            int64_t jy = (incy > 0 ? 0 : (-n + 1)*incy);
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * conj( y[jy] );
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    A(i,j) += x[ix] * tmp;
                    ix += incx;
                }
                jy += incy;
            }
        }
    }
    else {
        // RowMajor
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t i = 0; i < m; ++i) {
                // note: NOT skipping if x[i] is zero, for consistent NAN handling
                scalar_t tmp = alpha * x[i];
                for (int64_t j = 0; j < n; ++j) {
                    A(j,i) += tmp * conj( y[j] );
                }
            }
        }
        else if (incy == 1) {
            // x non-unit stride, y unit stride
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                scalar_t tmp = alpha * x[ix];
                for (int64_t j = 0; j < n; ++j) {
                    A(j,i) += tmp * conj( y[j] );
                }
                ix += incx;
            }
        }
        else {
            // x and y non-unit stride
            int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);
            int64_t ix = (incx > 0 ? 0 : (-m + 1)*incx);
            for (int64_t i = 0; i < m; ++i) {
                scalar_t tmp = alpha * x[ix];
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    A(j,i) += tmp * conj( y[jy] );
                    jy += incy;
                }
                ix += incx;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_GER_HH
