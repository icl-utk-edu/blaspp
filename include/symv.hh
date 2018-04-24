#ifndef BLAS_SYMV_HH
#define BLAS_SYMV_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup symv
inline
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    BLAS_ssymv( &uplo_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symv
inline
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    BLAS_dsymv( &uplo_, &n_,
               &alpha, A, &lda_, x, &incx_, &beta, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symv
inline
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    BLAS_csymv( &uplo_, &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda_,
                (blas_complex_float*) x, &incx_,
                (blas_complex_float*) &beta,
                (blas_complex_float*) y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup symv
inline
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Upper &&
                   uplo != Uplo::Lower );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
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
    BLAS_zsymv( &uplo_, &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda_,
                (blas_complex_double*) x, &incx_,
                (blas_complex_double*) &beta,
                (blas_complex_double*) y, &incy_ );
}

// =============================================================================
/// Symmetric matrix-vector multiply,
///     \f[ y = \alpha A x + \beta y, \f]
/// where alpha and beta are scalars, x and y are vectors,
/// and A is an n-by-n symmetric matrix.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///     Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///     What part of the matrix A is referenced,
///     the opposite triangle being assumed from symmetry.
///     - Uplo::Lower: only the lower triangular part of A is referenced.
///     - Uplo::Upper: only the upper triangular part of A is referenced.
///
/// @param[in] n
///     Number of rows and columns of the matrix A. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, A and x are not accessed.
///
/// @param[in] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1, n).
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in] beta
///     Scalar beta. If beta is zero, y need not be set on input.
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup symv

template< typename TA, typename TX, typename TY >
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    blas::scalar_type<TA, TX, TY> alpha,
    TA const *A, int64_t lda,
    TX const *x, int64_t incx,
    blas::scalar_type<TA, TX, TY> beta,
    TY *y, int64_t incy )
{
    typedef blas::scalar_type<TA, TX, TY> scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

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
                    tmp2 += A(i, j) * x[i];
                }
                y[j] += tmp1 * A(j, j) + alpha * tmp2;
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
                    tmp2 += A(i, j) * x[ix];
                    ix += incx;
                    iy += incy;
                }
                y[jy] += tmp1 * A(j, j) + alpha * tmp2;
                jx += incx;
                jy += incy;
            }
        }
    }
    else {
        // A is stored in lower triangle
        // form y += alpha * A * x
        if (incx == 1 && incy == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp1 = alpha*x[j];
                scalar_t tmp2 = zero;
                for (int64_t i = j+1; i < n; ++i) {
                    y[i] += tmp1 * A(i, j);
                    tmp2 += A(i, j) * x[i];
                }
                y[j] += tmp1 * A(j, j) + alpha * tmp2;
            }
        }
        else {
            // non-unit stride
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
                    tmp2 += A(i, j) * x[ix];
                }
                y[jy] += tmp1 * A(j, j) + alpha * tmp2;
                jx += incx;
                jy += incy;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYMV_HH
