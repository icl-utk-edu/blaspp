#ifndef BLAS_IAMAX_HH
#define BLAS_IAMAX_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
int64_t iamax(
    int64_t n,
    float const *x, int64_t incx )
{
    printf( "isamax implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(int64_t)) {
        throw_if_( n    > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_isamax( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
int64_t iamax(
    int64_t n,
    double const *x, int64_t incx )
{
    printf( "idamax implementation\n" );

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
    return f77_idamax( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
int64_t iamax(
    int64_t n,
    std::complex<float> const *x, int64_t incx )
{
    printf( "icamax implementation\n" );

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
    return f77_icamax( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
int64_t iamax(
    int64_t n,
    std::complex<double> const *x, int64_t incx )
{
    printf( "izamax implementation\n" );

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
    return f77_izamax( &n_, x, &incx_ );
}

// =============================================================================
/// @return Index of infinity-norm of vector, argmax_i |Re(x_i)| + |Im(x_i)|.
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
int64_t iamax(
    int64_t n,
    T const *x, int64_t incx )
{
    printf( "template iamax implementation\n" );

    typedef typename traits<T>::norm_t norm_t;
    
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail
    
    // todo: check NAN
    norm_t result = 0;
    int64_t index = 0;
    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            norm_t tmp = abs1( x[i] );
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
            norm_t tmp = abs1( x[ix] );
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
