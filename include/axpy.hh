#ifndef BLAS_AXPY_HH
#define BLAS_AXPY_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.
// We could use a template to avoid replicating argument checking code
// (see SLATE C++ API document), but then there are conflicts with the
// templated generic implementation below.

// -----------------------------------------------------------------------------
/// @ingroup axpy
inline
void axpy(
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_saxpy( &n_, &alpha, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup axpy
inline
void axpy(
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_daxpy( &n_, &alpha, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup axpy
inline
void axpy(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_caxpy( &n_,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) x, &incx_,
                (blas_complex_float*) y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup axpy
inline
void axpy(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );      // standard BLAS returns, doesn't fail
    blas_error_if( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    BLAS_zaxpy( &n_,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) x, &incx_,
                (blas_complex_double*) y, &incy_ );
}

// =============================================================================
/// Add scaled vector, \f$ y = \alpha x + y. \f$
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in] alpha
///     Scalar alpha. If alpha is zero, y is not updated.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in, out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup axpy

template< typename TX, typename TY >
void axpy(
    int64_t n,
    blas::scalar_type<TX, TY> alpha,
    TX const *x, int64_t incx,
    TY       *y, int64_t incy )
{
    typedef blas::scalar_type<TX, TY> scalar_t;

    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // quick return
    if (alpha == scalar_t(0))
        return;

    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            y[i] += alpha*x[i];
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] += alpha * x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_AXPY_HH
