#ifndef BLAS_GEMV_HH
#define BLAS_GEMV_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    if (layout == Layout::ColMajor)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m              > std::numeric_limits<blas_int>::max() );
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    f77_sgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
inline
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    if (layout == Layout::ColMajor)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m              > std::numeric_limits<blas_int>::max() );
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    f77_dgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
inline
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    if (layout == Layout::ColMajor)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m              > std::numeric_limits<blas_int>::max() );
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    f77_cgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x2, &incx_, &beta, y, &incy_ );

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
inline
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
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    if (layout == Layout::ColMajor)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( m              > std::numeric_limits<blas_int>::max() );
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    f77_zgemv( &trans_, &m_, &n_,
               &alpha, A, &lda_, x2, &incx_, &beta, y, &incy_ );

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

// =============================================================================
/// General matrix-vector multiply,
///     y = alpha*op(A)*x + beta*y,
/// where op(A) is one of
///     op(A) = A    or
///     op(A) = A^T  or
///     op(A) = A^H,
/// alpha and beta are scalars, x and y are vectors,
/// and A is an m-by-n matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///         Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] trans
///         The operation to be performed:
///         trans = Op::NoTrans   is y = alpha*A*x   + beta*y,
///         trans = Op::Trans     is y = alpha*A^T*x + beta*y,
///         trans = Op::ConjTrans is y = alpha*A^H*x + beta*y.
///
/// @param[in] m
///         Number of rows of the matrix A.
///
/// @param[in] n
///         Number of columns of the matrix A.
///
/// @param[in] alpha
///         Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///         The m-by-n matrix A.
///         ColMajor: stored in an lda-by-n array.
///         RowMajor: stored in an m-by-lda array.
///
/// @param[in] lda
///         Leading dimension of A.
///         ColMajor: lda >= max(1,m).
///         RowMajor: lda >= max(1,n).
///
/// @param[in] x
///         If trans = Op::NoTrans,
///                    the n-element vector x, of length (n-1)*abs(incx) + 1.
///         Otherwise, the m-element vector x, of length (m-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///         Scalar beta. When beta is zero, y need not be set on input.
///
/// @param[in,out] y
///         If trans = Op::NoTrans,
///                    the m-element vector y, of length (m-1)*abs(incy) + 1.
///         Otherwise, the n-element vector y, of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///         Stride between elements of y. incy must not be zero.
///         If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup blas2

template< typename TA, typename TX, typename TY >
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    typename blas::traits3<TA, TX, TY>::scalar_t alpha,
    TA const *A, int64_t lda,
    TX const *x, int64_t incx,
    typename blas::traits3<TA, TX, TY>::scalar_t beta,
    TY *y, int64_t incy )
{
    typedef typename blas::traits3<TA, TX, TY>::scalar_t scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( m < 0 );
    throw_if_( n < 0 );
    if (layout == Layout::ColMajor)
        throw_if_( lda < m );
    else
        throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // quick return
    if (m == 0 || n == 0 || (alpha == zero && beta == one))
        return;

    bool doconj = false;
    if (layout == Layout::RowMajor) {
        // A => A^T; A^T => A; A^H => A & conj
        std::swap( m, n );
        if (trans == Op::NoTrans) {
            trans = Op::Trans;
        }
        else {
            if (trans == Op::ConjTrans) {
                doconj = true;
            }
            trans = Op::NoTrans;
        }
    }

    int64_t lenx = (trans == Op::NoTrans ? n : m);
    int64_t leny = (trans == Op::NoTrans ? m : n);
    int64_t kx = (incx > 0 ? 0 : (-lenx + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-leny + 1)*incy);

    // ----------
    // form y = beta*y
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int64_t i = 0; i < leny; ++i) {
                    y[i] = zero;
                }
            }
            else {
                for (int64_t i = 0; i < leny; ++i) {
                    y[i] *= beta;
                }
            }
        }
        else {
            int64_t iy = ky;
            if (beta == zero) {
                for (int64_t i = 0; i < leny; ++i) {
                    y[iy] = zero;
                    iy += incy;
                }
            }
            else {
                for (int64_t i = 0; i < leny; ++i) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
    }
    if (alpha == zero)
        return;

    // ----------
    if (trans == Op::NoTrans && ! doconj) {
        // form y += alpha * A * x
        int64_t jx = kx;
        if (incy == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                for (int64_t i = 0; i < m; ++i) {
                    y[i] += tmp * A(i, j);
                }
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                int64_t iy = ky;
                for (int64_t i = 0; i < m; ++i) {
                    y[iy] += tmp * A(i, j);
                    iy += incy;
                }
            }
        }
    }
    else if (trans == Op::NoTrans && doconj) {
        // form y += alpha * conj( A ) * x
        // this occurs for row-major A^H * x
        int64_t jx = kx;
        if (incy == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                for (int64_t i = 0; i < m; ++i) {
                    y[i] += tmp * conj(A(i, j));
                }
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha*x[jx];
                jx += incx;
                int64_t iy = ky;
                for (int64_t i = 0; i < m; ++i) {
                    y[iy] += tmp * conj(A(i, j));
                    iy += incy;
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // form y += alpha * A^T * x
        int64_t jy = ky;
        if (incx == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += A(i, j) * x[i];
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += A(i, j) * x[ix];
                    ix += incx;
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
    }
    else {
        // form y += alpha * A^H * x
        int64_t jy = ky;
        if (incx == 1) {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += conj(A(i, j)) * x[i];
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
        else {
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = zero;
                int64_t ix = kx;
                for (int64_t i = 0; i < m; ++i) {
                    tmp += conj(A(i, j)) * x[ix];
                    ix += incx;
                }
                y[jy] += alpha*tmp;
                jy += incy;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_GEMV_HH
