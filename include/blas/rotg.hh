#ifndef BLAS_ROTG_HH
#define BLAS_ROTG_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Construct plane rotation that eliminates b, such that
//      [ z ] = [  c  s ] [ a ]
//      [ 0 ]   [ -s  c ] [ b ]
//
///     \f[ \begin{bmatrix} z     \\ 0      \end{bmatrix} =
///         \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
///         \begin{bmatrix} a     \\ b      \end{bmatrix} \f]
///
/// @see rot to apply the rotation.
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in, out] a
///     On entry, scalar a. On exit, set to z.
///
/// @param[in, out] b
///     On entry, scalar b. On exit, set to s, 1/c, or 0.
///
/// @param[out] c
///     Cosine of rotation; real.
///
/// @param[out] s
///     Sine of rotation; complex.
///
/// @ingroup rotg

template< typename TX, typename TY >
void rotg(
    TX a,
    TY b,
    blas::real_type<TX, TY>   c,
    blas::scalar_type<TX, TY> s )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTG_HH
