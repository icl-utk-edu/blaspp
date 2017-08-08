#ifndef BLAS_NRM2_HH
#define BLAS_NRM2_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
float nrm2(
    int64_t n,
    float const *x, int64_t incx )
{
    printf( "snrm2 implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_snrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
double nrm2(
    int64_t n,
    double const *x, int64_t incx )
{
    printf( "dnrm2 implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_dnrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
float nrm2(
    int64_t n,
    std::complex<float> const *x, int64_t incx )
{
    printf( "cnrm2 implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return f77_scnrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
double nrm2(
    int64_t n,
    std::complex<double> const *x, int64_t incx )
{
    printf( "znrm2 implementation\n" );

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
    return f77_dznrm2( &n_, x, &incx_ );
}

// =============================================================================
/// @return 2-norm of vector, || x ||.
///
/// Generic implementation for arbitrary data types.
/// TODO: does not currently scale to avoid over- or underflow.
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
nrm2(
    int64_t n,
    T const * x, int64_t incx )
{
    printf( "template nrm2 implementation\n" );

    typedef typename traits<T>::norm_t norm_t;

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: scale to avoid overflow & underflow
    norm_t result = 0;
    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += real(x[i]) * real(x[i]) + imag(x[i]) * imag(x[i]);
        }
    }
    else {
        // non-unit stride
        int64_t ix = 0;
        for (int64_t i = 0; i < n; ++i) {
            result += real(x[ix]) * real(x[ix]) + imag(x[ix]) * imag(x[ix]);
            ix += incx;
        }
    }
    return std::sqrt( result );
}

}  // namespace blas

#endif        //  #ifndef BLAS_NRM2_HH
