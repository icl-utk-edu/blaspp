#ifndef BLAS_DOT_HH
#define BLAS_DOT_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.
// Conjugated version, x^H y.

// -----------------------------------------------------------------------------
inline
float dot(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    return f77_sdot( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
inline
double dot(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    return f77_ddot( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
inline
std::complex<float> dot(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        f77_cdotc( &value, &n_, x, &incx_, y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        return f77_cdotc( &n_, x, &incx_, y, &incy_ );
    #endif
}

// -----------------------------------------------------------------------------
inline
std::complex<double> dot(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        f77_zdotc( &value, &n_, x, &incx_, y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        return f77_zdotc( &n_, x, &incx_, y, &incy_ );
    #endif
}

// =============================================================================
// Unconjugated version, x^T y

// -----------------------------------------------------------------------------
inline
float dotu(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
inline
double dotu(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
inline
std::complex<float> dotu(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        f77_cdotu( &value, &n_, x, &incx_, y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        return f77_cdotu( &n_, x, &incx_, y, &incy_ );
    #endif
}

// -----------------------------------------------------------------------------
inline
std::complex<double> dotu(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy )
{
    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        f77_zdotu( &value, &n_, x, &incx_, y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        return f77_zdotu( &n_, x, &incx_, y, &incy_ );
    #endif
}

// =============================================================================
/// @return dot product, x^H y.
/// @see dotu for unconjugated version, x^T y.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///         Number of elements in x and y.
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in,out] y
///         The n-element vector y, of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///         Stride between elements of y. incy must not be zero.
///         If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup blas1

template< typename TX, typename TY >
typename traits2<TX,TY>::scalar_t dot(
    int64_t n,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy )
{
    typedef typename traits2<TX,TY>::scalar_t scalar_t;

    // check arguments
    throw_if_( n < 0 );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    scalar_t result = 0;
    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += conj(x[i]) * y[i];
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            result += conj(x[ix]) * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return result;
}

// =============================================================================
/// @return unconjugated dot product, x^T y.
/// @see dot for conjugated version, x^H y.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///         Number of elements in x and y.
///
/// @param[in] x
///         The n-element vector x, of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @param[in,out] y
///         The n-element vector y, of length (n-1)*abs(incy) + 1.
///
/// @param[in] incy
///         Stride between elements of y. incy must not be zero.
///         If incy < 0, uses elements of y in reverse order: y(n-1), ..., y(0).
///
/// @ingroup blas1

template< typename TX, typename TY >
typename traits2<TX,TY>::scalar_t dotu(
    int64_t n,
    TX const *x, int64_t incx,
    TY const *y, int64_t incy )
{
    typedef typename traits2<TX,TY>::scalar_t scalar_t;

    // check arguments
    throw_if_( n < 0 );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    scalar_t result = 0;
    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            result += x[i] * y[i];
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            result += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }
    }
    return result;
}

}  // namespace blas

#endif        //  #ifndef BLAS_DOT_HH
