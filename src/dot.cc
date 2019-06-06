#include "blas_fortran.hh"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.
// Conjugated version, x^H y.

// -----------------------------------------------------------------------------
/// @ingroup dot
float dot(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy )
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
    return BLAS_sdot( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup dot
double dot(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy )
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
    return BLAS_ddot( &n_, x, &incx_, y, &incy_ );
}

// -----------------------------------------------------------------------------
/// @ingroup dot
std::complex<float> dot(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy )
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

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        BLAS_cdotc( (blas_complex_float*) &value, &n_,
                    (blas_complex_float*) x, &incx_,
                    (blas_complex_float*) y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        blas_complex_float value
             = BLAS_cdotc( &n_,
                           (blas_complex_float*) x, &incx_,
                           (blas_complex_float*) y, &incy_ );
        return *reinterpret_cast< std::complex<float>* >( &value );
    #endif
}

// -----------------------------------------------------------------------------
/// @ingroup dot
std::complex<double> dot(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy )
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

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        BLAS_zdotc( (blas_complex_double*) &value, &n_,
                    (blas_complex_double*) x, &incx_,
                    (blas_complex_double*) y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        blas_complex_double value
             = BLAS_zdotc( &n_,
                           (blas_complex_double*) x, &incx_,
                           (blas_complex_double*) y, &incy_ );
        return *reinterpret_cast< std::complex<double>* >( &value );
    #endif
}

// =============================================================================
// Unconjugated version, x^T y

// -----------------------------------------------------------------------------
/// @ingroup dotu
float dotu(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
double dotu(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy )
{
    return dot( n, x, incx, y, incy );
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
std::complex<float> dotu(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy )
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

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<float> value;
        BLAS_cdotu( (blas_complex_float*) &value, &n_,
                    (blas_complex_float*) x, &incx_,
                    (blas_complex_float*) y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        blas_complex_float value
             = BLAS_cdotu( &n_,
                           (blas_complex_float*) x, &incx_,
                           (blas_complex_float*) y, &incy_ );
        return *reinterpret_cast< std::complex<float>* >( &value );
    #endif
}

// -----------------------------------------------------------------------------
/// @ingroup dotu
std::complex<double> dotu(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy )
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

    #ifdef BLAS_COMPLEX_RETURN_ARGUMENT
        // Intel icc convention
        std::complex<double> value;
        BLAS_zdotu( (blas_complex_double*) &value, &n_,
                    (blas_complex_double*) x, &incx_,
                    (blas_complex_double*) y, &incy_ );
        return value;
    #else
        // GNU gcc convention
        blas_complex_double value
             = BLAS_zdotu( &n_,
                           (blas_complex_double*) x, &incx_,
                           (blas_complex_double*) y, &incy_ );
        return *reinterpret_cast< std::complex<double>* >( &value );
    #endif
}

}  // namespace blas
