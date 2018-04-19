#ifndef BLAS_NRM2_HH
#define BLAS_NRM2_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup nrm2
inline
float nrm2(
    int64_t n,
    float const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_snrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup nrm2
inline
double nrm2(
    int64_t n,
    double const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_dnrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup nrm2
inline
float nrm2(
    int64_t n,
    std::complex<float> const *x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( incx > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    return BLAS_scnrm2( &n_, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup nrm2
inline
double nrm2(
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
    return BLAS_dznrm2( &n_, x, &incx_ );
}

// =============================================================================
/// @return 2-norm of vector,
///     \f$ || x ||_2
///         = (\sum_{i=0}^{n-1} |x_i|^2)^{1/2}. \f$
///
/// Generic implementation for arbitrary data types.
/// TODO: generic implementation does not currently scale to avoid over- or underflow.
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
/// @ingroup nrm2

template< typename T >
real_type<T>
nrm2(
    int64_t n,
    T const * x, int64_t incx )
{
    typedef real_type<T> real_t;

    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    // todo: scale to avoid overflow & underflow
    real_t result = 0;
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
