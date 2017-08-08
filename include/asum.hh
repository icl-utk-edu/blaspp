#ifndef BLAS_ASUM_HH
#define BLAS_ASUM_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
float asum(
    int64_t n,
    float const *x, int64_t incx )
{
    printf( "sasum implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n    > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_sasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
double asum(
    int64_t n,
    double const *x, int64_t incx )
{
    printf( "dasum implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n    > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_dasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
float asum(
    int64_t n,
    std::complex<float> const *x, int64_t incx )
{
    printf( "scasum implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n    > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_scasum( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
double asum(
    int64_t n,
    std::complex<double> const *x, int64_t incx )
{
    printf( "dzasum implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n    > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_dzasum( &n_, x, &incx_ );
}

// =============================================================================
/// @return 1-norm of vector, || Re(x) ||_1 + || Im(x) ||_1.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///         Number of elements in x.
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*incx + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx > 0.
///
/// @ingroup blas1

template< typename T >
typename traits<T>::norm_t
asum(
    int64_t n,
    T const *x, int64_t incx )
{
    printf( "template asum implementation\n" );

    typedef typename traits<T>::norm_t norm_t;

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    norm_t result = 0;
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
