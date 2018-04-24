#ifndef BLAS_SCAL_HH
#define BLAS_SCAL_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup scal
inline
void scal(
    int64_t n,
    float alpha,
    float *x, int64_t incx )
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
    BLAS_sscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
inline
void scal(
    int64_t n,
    double alpha,
    double *x, int64_t incx )
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
    BLAS_dscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
inline
void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> *x, int64_t incx )
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
    BLAS_cscal( &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx_ );
}

// -----------------------------------------------------------------------------
/// @ingroup scal
inline
void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> *x, int64_t incx )
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
    BLAS_zscal( &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx_ );
}

// =============================================================================
/// Scale vector by constant, \f$ x = \alpha x. \f$
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*incx + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx > 0.
///
/// @ingroup scal

template< typename T >
void scal(
    int64_t n,
    T alpha,
    T* x, int64_t incx )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx <= 0 );  // standard BLAS returns, doesn't fail

    if (incx == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            x[i] *= alpha;
        }
    }
    else {
        // non-unit stride
        for (int64_t i = 0; i < n; i += incx) {
            x[i] *= alpha;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_SCAL_HH
