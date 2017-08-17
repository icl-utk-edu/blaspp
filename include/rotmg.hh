#ifndef BLAS_ROTMG_HH
#define BLAS_ROTMG_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>
#include <assert.h>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.
// Not available for complex.

// -----------------------------------------------------------------------------
inline
void rotmg(
    float *d1,
    float *d2,
    float *x1,
    float  y1,
    float  param[5] )
{
    f77_srotmg( d1, d2, x1, &y1, param );
}

// -----------------------------------------------------------------------------
inline
void rotmg(
    double *d1,
    double *d2,
    double *x1,
    double  y1,
    double  param[5] )
{
    f77_drotmg( d1, d2, x1, &y1, param );
}

// =============================================================================
/// Construct modified plane rotation.
/// TODO: describe.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in,out] d1
///         TODO: describe.
///
/// @param[in,out] d2
///         TODO: describe.
///
/// @param[in,out] x1
///         TODO: describe.
///
/// @param[in] dy1
///         TODO: describe.
///
/// @param[in] param
///         Array of length 5 giving parameters of modified plane rotation.
///         TODO: describe.
///
/// @ingroup blas1

template< typename T >
void rotmg(
    T *d1,
    T *d2,
    T *x1,
    T  y1,
    T  param[5] )
{
    assert( false );
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTMG_HH
