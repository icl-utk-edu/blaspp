// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTMG_HH
#define BLAS_ROTMG_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Construct modified (fast) plane rotation, H, that eliminates b, such that
/// \[
///       \begin{bmatrix} z \\ 0 \end{bmatrix}
///     = H
///       \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
///       \begin{bmatrix} a \\ b \end{bmatrix}.
/// \]
///
/// @see rotm to apply the rotation.
///
/// With modified plane rotations, vectors u and v are held in factored form as
/// \[
///     \begin{bmatrix} u^T \\ v^T \end{bmatrix} =
///     \begin{bmatrix} \sqrt{d_1} & 0 \\ 0 & \sqrt{d_2} \end{bmatrix}
///     \begin{bmatrix} x^T \\ y^T \end{bmatrix}.
/// \]
///
/// Application of H to vectors x and y requires 4n flops (2n mul, 2n add)
/// instead of 6n flops (4n mul, 2n add) as in standard plane rotations.
///
/// Let param = [ flag, $h_{11}, h_{21}, h_{12}, h_{22}$ ].
/// Then H has one of the following forms:
///
/// - For flag = -1,
///     \[
///         H = \begin{bmatrix}
///             h_{11}  &  h_{12}
///         \\  h_{21}  &  h_{22}
///         \end{bmatrix}
///     \]
///
/// - For flag = 0,
///     \[
///         H = \begin{bmatrix}
///             1       &  h_{12}
///         \\  h_{21}  &  1
///         \end{bmatrix}
///     \]
///
/// - For flag = 1,
///     \[
///         H = \begin{bmatrix}
///             h_{11}  &  1
///         \\  -1      &  h_{22}
///         \end{bmatrix}
///     \]
///
/// - For flag = -2,
///     \[
///         H = \begin{bmatrix}
///             1  &  0
///         \\  0  &  1
///         \end{bmatrix}
///     \]
///
/// Generic implementation for arbitrary data types.
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

template <typename T>
void rotmg(
    T *d1,
    T *d2,
    T *a,
    T  b,
    T  param[5] )
{
    using std::abs;

    // Constants
    const T zero = 0;
    const T one = 1;
    const T gam = 4096;
    const T gamsq = gam*gam;
    const T rgamsq = one/gamsq;

    T& x1 = *a;
    T& y1 = b;

    T h11 = zero;
    T h12 = zero;
    T h21 = zero;
    T h22 = zero;

    if (*d1 < zero) {
        param[0] = -1;
        *d1 = zero;
        *d2 = zero;
        x1 = zero;
    }
    else {
        T p2 = (*d2)*y1;
        if (p2 == zero) {
            param[0] = -2;
            return;
        }

        T p1 = (*d1)*x1;
        T q2 = p2*y1;
        T q1 = p1*x1;

        if (abs(q1) > abs(q2)) {
            param[0] = zero;
            h21 = -y1/x1;
            h12 = p2/p1;
            T u = one - h12*h21;
            if (u > zero) {
                *d1 /= u;
                *d2 /= u;
                x1 *= u;
            }
        }
        else if (q2 < zero) {
            param[0] = -1;
            *d1 = zero;
            *d2 = zero;
            x1 = zero;
        }
        else {
            param[0] = 1;
            h11 = p1/p2;
            h22 = x1/y1;
            T u = one + h11*h22;
            T stemp = *d2/u;
            *d2 = *d1/u;
            *d1 = stemp;
            x1 = y1*u;
        }

        if (*d1 != zero) {
            while ((*d1 <= rgamsq) || (*d1 >= gamsq)) {
                if (param[0] == 0) {
                    h11 = one;
                    h22 = one;
                    param[0] = -1;
                }
                else {
                    h21 = -one;
                    h12 = one;
                    param[0] = -1;
                }
                if (*d1 <= rgamsq) {
                    *d1 *= gam*gam;
                    x1 /= gam;
                    h11 /= gam;
                    h12 /= gam;
                }
                else {
                    *d1 /= gam*gam;
                    x1 *= gam;
                    h11 *= gam;
                    h12 *= gam;
                }
            }
        }

        if (*d2 != zero) {
            while ((abs(*d2) <= rgamsq) || (abs(*d2) >= gamsq)) {
                if (param[0] == 0) {
                    h11=one;
                    h22=one;
                    param[0]=-1;
                }
                else {
                    h21=-one;
                    h12=one;
                    param[0]=-1;
                }
                if (abs(*d2) <= rgamsq) {
                    *d2 *= gam*gam;
                    h21 /= gam;
                    h22 /= gam;
                }
                else {
                    *d2 /= gam*gam;
                    h21 *= gam;
                    h22 *= gam;
                }
            }
        }
    }

    if (param[0] < 0) {
        param[1] = h11;
        param[2] = h21;
        param[3] = h12;
        param[4] = h22;
    }
    else if (param[0] == 0) {
        param[2] = h21;
        param[3] = h12;
    }
    else {
        param[1] = h11;
        param[4] = h22;
    }
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTMG_HH
