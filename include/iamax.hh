#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup iamax
inline
int64_t iamax(
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
    return BLAS_isamax( &n_, x, &incx_ ) - 1;
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
inline
int64_t iamax(
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
    return BLAS_idamax( &n_, x, &incx_ ) - 1;
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
inline
int64_t iamax(
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
    return BLAS_icamax( &n_, x, &incx_ ) - 1;
}

// -----------------------------------------------------------------------------
/// @ingroup iamax
inline
int64_t iamax(
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
    return BLAS_izamax( &n_, x, &incx_ ) - 1;
}

// =============================================================================
/// @return Index of infinity-norm of vector,
///     \f$ || x ||_{inf}
///         = \text{argmax}_{i=0}^{n-1} |Re(x_i)| + |Im(x_i)|. \f$
///     Returns -1 if n = 0.
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
/// @ingroup iamax

template< typename T >
int64_t iamax(
    int64_t n,
    T const *x, int64_t incx )
{
    typedef real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: check NAN
    real_t result = -1;
    int64_t index = -1;
    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            real_t tmp = abs1( x[i] );
            if (tmp > result) {
                result = tmp;
                index = i;
            }
        }
    }
    else {
        // non-unit stride
        int64_t ix = 0;
        for (int64_t i = 0; i < n; ++i) {
            real_t tmp = abs1( x[ix] );
            if (tmp > result) {
                result = tmp;
                index = i;
            }
            ix += incx;
        }
    }
    return index;
}

}  // namespace blas

#endif        //  #ifndef BLAS_IAMAX_HH
