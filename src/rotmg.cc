#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.
// Not available for complex.

// -----------------------------------------------------------------------------
/// @ingroup rotmg
void rotmg(
    float *d1,
    float *d2,
    float *a,
    float  b,
    float  param[5] )
{
    BLAS_srotmg( d1, d2, a, &b, param );
}

// -----------------------------------------------------------------------------
/// @ingroup rotmg
void rotmg(
    double *d1,
    double *d2,
    double *a,
    double  b,
    double  param[5] )
{
    BLAS_drotmg( d1, d2, a, &b, param );
}

}  // namespace blas
