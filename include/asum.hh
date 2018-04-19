#ifndef BLAS_ASUM_HH
#define BLAS_ASUM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup asum
inline
float asum(
    int64_t n,
    float const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_sasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup asum
inline
double asum(
    int64_t n,
    double const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_dasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup asum
inline
float asum(
    int64_t n,
    std::complex<float> const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_scasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup asum
inline
double asum(
    int64_t n,
    std::complex<double> const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n    > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_dzasum( &n_, x, &incx_ );
}

// =============================================================================
/// @return 1-norm of vector,
///     \f$ || Re(x) ||_1 + || Im(x) ||_1
///         = \sum_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|. \f$
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup asum

template< typename T >
real_type<T>
asum(
    int64_t n,
    T const *x, int64_t incx )
{
    typedef real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    real_t result = 0;
    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += abs1( x[i] );
        }
    }
    else {
        // non-unit stride
        int64_t ix = 0;
        for (int64_t i = 0; i < n; ++i) {
            result += abs1( x[ix] );
            ix += incx;
        }
    }
    return result;
}

}  // namespace blas

#endif        //  #ifndef BLAS_ASUM_HH
