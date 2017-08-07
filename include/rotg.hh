#ifndef BLAS_ROTG_HH
#define BLAS_ROTG_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>
#include <assert.h>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void rotg(
    float *a,
    float *b,
    float *c,
    float *s )
{
    printf( "srotg implementation\n" );
    f77_srotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
inline
void rotg(
    double *a,
    double *b,
    double *c,
    double *s )
{
    printf( "drotg implementation\n" );
    f77_drotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
inline
void rotg(
    std::complex<float> *a,
    std::complex<float> *b,  // const in BLAS implementation, oddly
    float *c,
    std::complex<float> *s )
{
    printf( "crotg implementation\n" );
    f77_crotg( a, b, c, s );
}

// -----------------------------------------------------------------------------
inline
void rotg(
    std::complex<double> *a,
    std::complex<double> *b,  // const in BLAS implementation, oddly
    double *c,
    std::complex<double> *s )
{
    printf( "zrotg implementation\n" );
    f77_zrotg( a, b, c, s );
}

// =============================================================================
/// Construct plane rotation.
/// TODO: describe.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in,out] a
///         Scalar a.
///
/// @param[in,out] b
///         Scalar b.
///
/// @param[in] c
///         Cosine of rotation; real.
///
/// @param[in] s
///         Sine of rotation; complex.
///
/// @ingroup blas1

template< typename TX, typename TY >
void rotg(
    TX a,
    TY b,
    typename blas::traits2<TX,TY>::norm_t   c,
    typename blas::traits2<TX,TY>::scalar_t s )
{
    printf( "template rotg implementation\n" );

    assert( false );
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTG_HH
