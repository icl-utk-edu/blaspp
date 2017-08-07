#ifndef BLAS_ROT_HH
#define BLAS_ROT_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>
#include <assert.h>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void rot(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy,
    float c,
    float s )
{
    printf( "srot implementation\n" );

    // check arguments
    throw_if_( n < 0 );      // standard BLAS returns, doesn't fail
    throw_if_( incx == 0 );  // standard BLAS doesn't detect inc[xy] == 0
    throw_if_( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(int64_t)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;
    f77_srot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// -----------------------------------------------------------------------------
inline
void rot(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy,
    double c,
    double s )
{
    printf( "drot implementation\n" );

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
    f77_drot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// -----------------------------------------------------------------------------
inline
void rot(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy,
    float c,
    std::complex<float> s )
{
    printf( "crot implementation\n" );

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
    f77_crot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// -----------------------------------------------------------------------------
inline
void rot(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy,
    double c,
    std::complex<double> s )
{
    printf( "zrot implementation\n" );

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
    f77_zrot( &n_, x, &incx_, y, &incy_, &c, &s );
}

// =============================================================================
///                       [ x^T ]   [  c  s ] [ x^T ]
/// Apply plane rotation, [ y^T ] = [ -s  c ] [ y^T ].
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] n
///         Number of elements in x and y.
///
/// @param[in,out] x
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
/// @param[in] c
///         Cosine of rotation; real.
///
/// @param[in] s
///         Sine of rotation; complex.
///
/// @ingroup blas1

template< typename TX, typename TY >
void rot(
    int64_t n,
    TX *x, int64_t incx,
    TY *y, int64_t incy,
    typename blas::traits2<TX,TY>::norm_t   c,
    typename blas::traits2<TX,TY>::scalar_t s )
{
    printf( "template rot implementation\n" );

    typedef typename blas::traits2<TX,TY>::scalar_t scalar_t;

    // check arguments
    throw_if_( n < 0 );
    throw_if_( incx == 0 );
    throw_if_( incy == 0 );

    if (incx == 1 && incy == 1) {
        // unit stride
        for (int64_t i = 0; i < n; ++i) {
            assert( false );
        }
    }
    else {
        // non-unit stride
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            assert( false );
            ix += incx;
            iy += incy;
        }
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROT_HH
