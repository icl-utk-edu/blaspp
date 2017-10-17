#ifndef BLAS_SCAL_HH
#define BLAS_SCAL_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void scal(
    int64_t n,
    float alpha,
    float *x, int64_t incx )
{
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
    f77_sscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
void scal(
    int64_t n,
    double alpha,
    double *x, int64_t incx )
{
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
    f77_dscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> *x, int64_t incx )
{
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
    f77_cscal( &n_, &alpha, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> *x, int64_t incx )
{
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
    f77_zscal( &n_, &alpha, x, &incx_ );
}

// =============================================================================
/// Scale vector by constant, x = alpha*x.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///         Number of elements in x.
///
/// @param[in] alpha
///         Scalar alpha.
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*incx + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx > 0.
///
/// @ingroup blas1

template< typename T >
void scal(
    int64_t n,
    T alpha,
    T* x, int64_t incx )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx <= 0 );  // standard BLAS returns, doesn't fail

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
