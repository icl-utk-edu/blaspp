#ifndef BLAS_HEMV_HH
#define BLAS_HEMV_HH

#include "blas_fortran.hh"
#include "blas_util.hh"
#include "symv.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy )
{
    symv( layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy );
}

// -----------------------------------------------------------------------------
inline
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy )
{
    symv( layout, uplo, n, alpha, A, lda, x, incx, beta, y, incy );
}

// -----------------------------------------------------------------------------
inline
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy )
{
    printf( "chemv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Upper &&
               uplo != Uplo::Lower );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    // if x2=x, then it isn't modified
    std::complex<float> *x2 = const_cast< std::complex<float>* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate alpha, beta, x (in x2), and y (in-place)
        alpha = conj( alpha );
        beta  = conj( beta );

        x2 = new std::complex<float>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
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

    char uplo_ = uplo2char( uplo );
    f77_chemv( &uplo_, &n_,
               &alpha, A, &lda_, x2, &incx_, &beta, y, &incy_ );

    if (layout == Layout::RowMajor) {
        // y = conj( y )
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = conj( y[iy] );
            iy += incy;
        }
    }
}

// -----------------------------------------------------------------------------
inline
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy )
{
    printf( "zhemv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Upper &&
               uplo != Uplo::Lower );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    // if x2=x, then it isn't modified
    std::complex<double> *x2 = const_cast< std::complex<double>* >( x );
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate alpha, beta, x (in x2), and y (in-place)
        alpha = conj( alpha );
        beta  = conj( beta );

        x2 = new std::complex<double>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
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

    char uplo_ = uplo2char( uplo );
    f77_zhemv( &uplo_, &n_,
               &alpha, A, &lda_, x2, &incx_, &beta, y, &incy_ );

    if (layout == Layout::RowMajor) {
        // y = conj( y )
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = conj( y[iy] );
            iy += incy;
        }
    }
}

// =============================================================================
/// Symmetric matrix-vector multiply,
///     y = alpha*A*x + beta*y,
/// where alpha and beta are scalars, x and y are vectors,
/// and A is an n-by-n symmetric matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///         Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///         TODO
///
/// @param[in] n
///         Number of rows and columns of the matrix A.
///
/// @param[in] alpha
///         Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///         The n-by-n matrix A, stored in an lda-by-n array.
///         TODO uplo
///
/// @param[in] lda
///         Leading dimension of A, i.e., column stride. lda >= max(1,n).
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///         Scalar beta. When beta is zero, y need not be set on input.
///
/// @param[in,out] y
///         The n-element vector y, of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///         Stride between elements of y. incy must not be zero.
///         If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup blas2

template< typename TA, typename TX, typename TY >
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    typename blas::traits3<TA, TX, TY>::scalar_t alpha,
    TA const *A, int64_t lda,
    TX const *x, int64_t incx,
    typename blas::traits3<TA, TX, TY>::scalar_t beta,
    TY *y, int64_t incy )
{
    printf( "template hemv implementation\n" );

    typedef typename blas::traits3<TA, TX, TY>::scalar_t scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    // quick return
    if (n == 0 || (alpha == zero && beta == one))
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);

    // form y = beta*y
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) {
                for (int64_t i = 0; i < n; ++i) {
                    y[i] = zero;
                }
            }
            else {
                for (int64_t i = 0; i < n; ++i) {
                    y[i] *= beta;
                }
            }
        }
        else {
            int64_t iy = ky;
            if (beta == zero) {
                for (int64_t i = 0; i < n; ++i) {
                    y[iy] = zero;
                    iy += incy;
                }
            }
            else {
                for (int64_t i = 0; i < n; ++i) {
                    y[iy] *= beta;
                    iy += incy;
                }
            }
        }
    }
    if (alpha == zero)
        return;

    if (layout == Layout::ColMajor) {
        if (uplo == Uplo::Upper) {
            // A is stored in upper triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = 0; i < j; ++i) {
                        y[i] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = kx;
                    int64_t iy = ky;
                    for (int64_t i = 0; i < j; ++i) {
                        y[iy] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[ix];
                        ix += incx;
                        iy += incy;
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else if (uplo == Uplo::Lower) {
            // A is stored in lower triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = j+1; i < n; ++i) {
                        y[i] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = jx;
                    int64_t iy = jy;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        iy += incy;
                        y[iy] += tmp1 * A(i, j);
                        tmp2 += conj( A(i, j) ) * x[ix];
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }
    else {
        if (uplo == Uplo::Lower) {
            // A is stored in lower triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = 0; i < j; ++i) {
                        y[i] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = kx;
                    int64_t iy = ky;
                    for (int64_t i = 0; i < j; ++i) {
                        y[iy] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[ix];
                        ix += incx;
                        iy += incy;
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
        else if (uplo == Uplo::Upper) {
            // A is stored in upper triangle
            // form y += alpha * A * x
            if (incx == 1 && incy == 1) {
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[j];
                    scalar_t tmp2 = zero;
                    for (int64_t i = j+1; i < n; ++i) {
                        y[i] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[i];
                    }
                    y[j] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                }
            }
            else {
                int64_t jx = kx;
                int64_t jy = ky;
                for (int64_t j = 0; j < n; ++j) {
                    scalar_t tmp1 = alpha*x[jx];
                    scalar_t tmp2 = zero;
                    int64_t ix = jx;
                    int64_t iy = jy;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        iy += incy;
                        y[iy] += tmp1 * conj( A(i, j) );
                        tmp2 += A(i, j) * x[ix];
                    }
                    y[jy] += tmp1 * real( A(j, j) ) + alpha * tmp2;
                    jx += incx;
                    jy += incy;
                }
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_HEMV_HH
