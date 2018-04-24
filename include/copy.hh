#ifndef BLAS_COPY_HH
#define BLAS_COPY_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup copy
inline
void copy(
    int64_t n,
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
    BLAS_scopy( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup copy
inline
void copy(
    int64_t n,
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
    BLAS_dcopy( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup copy
inline
void copy(
    int64_t n,
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
    BLAS_ccopy( &n_,
                (blas_complex_float*) x, &incx_,
                (blas_complex_float*) y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup copy
inline
void copy(
    int64_t n,
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
    BLAS_zcopy( &n_,
                (blas_complex_double*) x, &incx_,
                (blas_complex_double*) y, &incy_ );
}

// =============================================================================
/// Copy vector, \f$ y = x. \f$
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///     Number of elements in x and y. n >= 0.
///
/// @param[in] x
///     The n-element vector x, in an array of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///     Stride between elements of x. incx must not be zero.
///     If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[out] y
///     The n-element vector y, in an array of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///     Stride between elements of y. incy must not be zero.
///     If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup copy

template< typename TX, typename TY >
void copy(
    int64_t n,
    TX const *x, int64_t incx,
    TY       *y, int64_t incy )
{
    // check arguments
    blas_error_if( n < 0 );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            y[i] = x[i];
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            y[iy] = x[ix];
            ix += incx;
            iy += incy;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_COPY_HH
