#ifndef BLAS_HER2_HH
#define BLAS_HER2_HH

#include "blas_fortran.hh"
#include "blas_util.hh"
#include "syr2.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda )
{
    syr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda )
{
    syr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

// -----------------------------------------------------------------------------
inline
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda )
{
    printf( "cher2 implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
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

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    f77_cher2( &uplo_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
}

// -----------------------------------------------------------------------------
inline
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda )
{
    printf( "zher2 implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
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

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    f77_zher2( &uplo_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
}

// =============================================================================
/// Symmetric matrix rank-2 update,
///     A = alpha*x*y^H + alpha*y*x^H + A,
/// where alpha is a scalar, x and y are vectors,
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
///         Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in,out] y
///         The n-element vector y, of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///         Stride between elements of y. incy must not be zero.
///         If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @param[in] A
///         The n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///         Leading dimension of A, i.e., column stride. lda >= max(1,n).
///
/// @ingroup blas2

template< typename TA, typename TX, typename TY >
void her2(
    blas::Layout layout,
    blas::Uplo  uplo,
    int64_t n,
    typename blas::traits3<TA, TX, TY>::scalar_t alpha,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy,
    TA *A, int64_t lda )
{
    printf( "template her2 implementation\n" );

    typedef typename blas::traits3<TA, TX, TY>::scalar_t scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( n < 0 );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );
    throw_if_( lda < n );

    // quick return
    if (n == 0 || alpha == zero)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    int64_t ky = (incy > 0 ? 0 : (-n + 1)*incy);
    if (uplo == Uplo::Upper) {
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] or y[j] is zero, for consistent NAN handling
                scalar_t tmp1 = alpha * conj( y[j] );
                scalar_t tmp2 = conj( alpha * x[j] );
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i,j) += x[i]*tmp1 + y[i]*tmp2;
                }
                A(j,j) = real( A(j,j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            int64_t jy = ky;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[jy] );
                scalar_t tmp2 = conj( alpha * x[jx] );
                int64_t ix = kx;
                int64_t iy = ky;
                for (int64_t i = 0; i <= j-1; ++i) {
                    A(i,j) += x[ix]*tmp1 + y[iy]*tmp2;
                    ix += incx;
                    iy += incy;
                }
                A(j,j) = real( A(j,j) ) + real( x[jx]*tmp1 + y[jy]*tmp2 );
                jx += incx;
                jy += incy;
            }
        }
    }
    else {
        // lower triangle
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[j] );
                scalar_t tmp2 = conj( alpha * x[j] );
                A(j,j) = real( A(j,j) ) + real( x[j]*tmp1 + y[j]*tmp2 );
                for (int64_t i = j+1; i < n; ++i) {
                    A(i,j) += x[i]*tmp1 + y[i]*tmp2;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            int64_t jy = ky;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha * conj( y[jy] );
                scalar_t tmp2 = conj( alpha * x[jx] );
                A(j,j) = real( A(j,j) ) + real( x[jx]*tmp1 + y[jy]*tmp2 );
                int64_t ix = jx;
                int64_t iy = jy;
                for (int64_t i = j+1; i < n; ++i) {
                    ix += incx;
                    iy += incy;
                    A(i,j) += x[ix]*tmp1 + y[iy]*tmp2;
                }
                jx += incx;
                jy += incy;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_HER2_HH
