#ifndef BLAS_ROTMG_HH
#define BLAS_ROTMG_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Construct modified (fast) plane rotation, H, that eliminates b, such that
//      [ z ] = H [ sqrt(d1)    0  ] [ a ]
//      [ 0 ]     [  0    sqrt(d2) ] [ b ]
//
///     \f[ \begin{bmatrix} z \\ 0 \end{bmatrix} =
///         H
///         \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
///         \begin{bmatrix} a \\ b \end{bmatrix} \f]
///
/// @see rotm to apply the rotation.
///
/// With modified plane rotations, vectors u and v are held in factored form as
//      [ u^T ] = [ sqrt(d1)    0  ] [ x^T ]
//      [ v^T ] = [  0    sqrt(d2) ] [ y^T ]
///
///     \f[ \begin{bmatrix} u^T \\ v^T \end{bmatrix} =
///         \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
///         \begin{bmatrix} x^T \\ y^T \end{bmatrix}. \f]
///
/// Application of H to vectors x and y requires 4n flops (2n mul, 2n add)
/// instead of 6n flops (4n mul, 2n add) as in standard plane rotations.
///
/// Let param = [ flag, \f$ h_{11}, h_{21}, h_{12}, h_{22} \f$ ].
/// Then H has one of the following forms:
///
/// - For flag = -1,
///     \f[ H = \begin{bmatrix}
///         h_{11}  &  h_{12}
///     \\  h_{21}  &  h_{22}
///     \end{bmatrix} \f]
///
/// - For flag = 0,
///     \f[ H = \begin{bmatrix}
///         1       &  h_{12}
///     \\  h_{21}  &  1
///     \end{bmatrix} \f]
///
/// - For flag = 1,
///     \f[ H = \begin{bmatrix}
///         h_{11}  &  1
///     \\  -1      &  h_{22}
///     \end{bmatrix} \f]
///
/// - For flag = -2,
///     \f[ H = \begin{bmatrix}
///         1  &  0
///     \\  0  &  1
///     \end{bmatrix} \f]
///
/// Generic implementation for arbitrary data types.
/// TODO: generic version not yet implemented.
///
/// @param[in, out] d1
///     sqrt(d1) is scaling factor for vector x.
///
/// @param[in, out] d2
///     sqrt(d2) is scaling factor for vector y.
///
/// @param[in, out] a
///     On entry, scalar a. On exit, set to z.
///
/// @param[in] b
///     On entry, scalar b.
///
/// @param[out] param
///     Array of length 5 giving parameters of modified plane rotation,
///     as described above.
///
/// __Further details__
///
/// Hammarling, Sven. A note on modifications to the Givens plane rotation.
/// IMA Journal of Applied Mathematics, 13:215-218, 1974.
/// http://dx.doi.org/10.1093/imamat/13.2.215
/// (Note the notation swaps u <=> x, v <=> y, d_i -> l_i.)
///
/// @ingroup rotmg

template< typename T >
void rotmg(
    T *d1,
    T *d2,
    T *a,
    T  b,
    T  param[5] )
{
    throw std::exception();  // not yet implemented
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTMG_HH
