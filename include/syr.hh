#ifndef BLAS_SYR_HH
#define BLAS_SYR_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup syr
inline
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *A, int64_t lda )
{
    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    f77_ssyr( &uplo_, &n_, &alpha, x, &incx_, A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syr
inline
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *A, int64_t lda )
{
    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    f77_dsyr( &uplo_, &n_, &alpha, x, &incx_, A, &lda_ );
}

// [cz]syr are in LAPACK, so those wrappers are in LAPACK++

// =============================================================================
/// Symmetric matrix rank-1 update,
///     \f[ A = \alpha x x^T + A, \f]
/// where alpha is a scalar, x is a vector,
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
///     Scalar alpha. If alpha is zero, A is not updated.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in,out] A
///     The n-by-n matrix A, stored in an lda-by-n array [RowMajor: n-by-lda].
///
/// @param[in] lda
///     Leading dimension of A. lda >= max(1,n).
///
/// @ingroup syr

template< typename TA, typename TX >
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    typename blas::traits2<TA, TX>::scalar_t alpha,
    TX const *x, int64_t incx,
    TA       *A, int64_t lda )
{
    printf( "template syr implementation\n" );

    typedef typename blas::traits2<TA, TX>::scalar_t scalar_t;

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
    throw_if_( lda < n );

    // quick return
    if (n == 0 || alpha == zero)
        return;

    // for row major, swap lower <=> upper
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);
    if (uplo == Uplo::Upper) {
        if (incx == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                // note: NOT skipping if x[j] is zero, for consistent NAN handling
                scalar_t tmp = alpha * x[j];
                for (int64_t i = 0; i <= j; ++i) {
                    A(i,j) += x[i] * tmp;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[jx];
                int64_t ix = kx;
                for (int64_t i = 0; i <= j; ++i) {
                    A(i,j) += x[ix] * tmp;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }
    else {
        // lower triangle
        if (incx == 1) {
            // unit stride
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[j];
                for (int64_t i = j; i < n; ++i) {
                    A(i,j) += x[i] * tmp;
                }
            }
        }
        else {
            // non-unit stride
            int64_t jx = kx;
            for (int64_t j = 0; j < n; ++j) {
                scalar_t tmp = alpha * x[jx];
                int64_t ix = jx;
                for (int64_t i = j; i < n; ++i) {
                    A(i,j) += x[ix] * tmp;
                    ix += incx;
                }
                jx += incx;
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_SYR_HH
