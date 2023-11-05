// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_ROTG_HH
#define BLAS_ROTG_HH

#include "blas/util.hh"

#include <limits>

namespace blas {

// =============================================================================
/// Construct plane rotation that eliminates b, such that:
/// \[
///       \begin{bmatrix} r     \\ 0      \end{bmatrix}
///     = \begin{bmatrix} c & s \\ -s & c \end{bmatrix}
///       \begin{bmatrix} a     \\ b      \end{bmatrix}.
/// \]
///
/// @see rot to apply the rotation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in, out] a
///     On entry, scalar a. On exit, set to r.
///
/// @param[in, out] b
///     On entry, scalar b. On exit, set to s, 1/c, or 0.
///
/// @param[out] c
///     Cosine of rotation; real.
///
/// @param[out] s
///     Sine of rotation; real.
///
/// __Further details__
///
/// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
/// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
///
/// @ingroup rotg

template <typename TA, typename TB>
void rotg(
    TA *a,
    TB *b,
    blas::real_type<TA, TB> *c,
    blas::real_type<TA, TB> *s )
{
    typedef real_type<TA, TB> real_t;
    using std::abs;

    // Constants
    const real_t one  = 1;
    const real_t zero = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();

    // Norms
    const real_t anorm = abs(*a);
    const real_t bnorm = abs(*b);

    // quick return
    if (bnorm == zero) {
        *c = one;
        *s = zero;
        *b = TB( 0.0 );
    }
    else if (anorm == zero) {
        *c = zero;
        *s = one;
        *a = *b;
        *b = TB( 1.0 );
    }
    else {
        real_t scl = min( safmax, max(safmin, anorm, bnorm) );
        real_t sigma = (anorm > bnorm)
            ? sgn(*a)
            : sgn(*b);
        real_t r = sigma * scl * sqrt( (*a/scl) * (*a/scl) + (*b/scl) * (*b/scl) );
        *c = *a / r;
        *s = *b / r;
        *a = r;
        if (anorm > bnorm)
            *b = *s;
        else if (*c != zero)
            *b = one / *c;
        else
            *b = one;
    }
}

// =============================================================================
/// Construct plane rotation that eliminates b, such that:
/// \[
///       \begin{bmatrix} r     \\ 0      \end{bmatrix}
///     = \begin{bmatrix} c & s \\ -conjg(s) & c \end{bmatrix}
///       \begin{bmatrix} a     \\ b      \end{bmatrix}.
/// \]
///
/// @see rot to apply the rotation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in, out] a
///     On entry, scalar a. On exit, set to r.
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
/// __Further details__
///
/// Anderson E (2017) Algorithm 978: Safe scaling in the level 1 BLAS.
/// ACM Trans Math Softw 44:. https://doi.org/10.1145/3061665
///
/// @ingroup rotg

template <typename TA, typename TB>
void rotg(
    std::complex<TA> *a,
    std::complex<TB> *b,
    blas::real_type<TA, TB>   *c,
    blas::complex_type<TA, TB> *s )
{
    typedef real_type<TA, TB> real_t;
    typedef complex_type<TA, TB> scalar_t;
    using std::abs;

    #define BLAS_ABSSQ(t_) real(t_)*real(t_) + imag(t_)*imag(t_)

    // Constants
    const real_t r_one = 1;
    const real_t r_zero = 0;
    const scalar_t zero = 0;
    const std::complex<TA> zero_ta = 0;
    const std::complex<TB> zero_tb = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();
    const real_t rtmin = root_min<real_t>();
    const real_t rtmax = root_max<real_t>();

    // quick return
    if (*b == zero_tb) {
        *c = r_one;
        *s = zero;
        return;
    }

    if (*a == zero_ta) {
        *c = r_zero;
        real_t g1 = max( abs(real(*b)), abs(imag(*b)) );
        if (g1 > rtmin && g1 < rtmax) {
            // Use unscaled algorithm
            real_t g2 = BLAS_ABSSQ( *b );
            real_t d = sqrt( g2 );
            *s = conj( *b ) / d;
            *a = d;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = (*b)*uu;
            real_t g2 = BLAS_ABSSQ( gs );
            real_t d = sqrt( g2 );
            *s = conj( gs ) / d;
            *a = d*u;
        }
    }
    else {
        real_t f1 = max( abs(real(*a)), abs(imag(*a)) );
        real_t g1 = max( abs(real(*b)), abs(imag(*b)) );
        if (f1 > rtmin && f1 < rtmax
            && g1 > rtmin && g1 < rtmax) {
            // Use unscaled algorithm
            real_t f2 = BLAS_ABSSQ( *a );
            real_t g2 = BLAS_ABSSQ( *b );
            real_t h2 = f2 + g2;
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            *c  = f2*p;
            *s  = conj( *b )*( (*a)*p );
            *a *= h2*p;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, f1, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = (*b)*uu;
            real_t g2 = BLAS_ABSSQ( gs );
            real_t f2, h2, w;
            scalar_t fs;
            if (f1*uu < rtmin) {
                // a is not well-scaled when scaled by g1.
                real_t v = min( safmax, max( safmin, f1 ) );
                real_t vv = r_one / v;
                w = v * uu;
                fs = (*a)*vv;
                f2 = BLAS_ABSSQ( fs );
                h2 = f2*w*w + g2;
            }
            else {
                // Otherwise use the same scaling for a and b.
                w = r_one;
                fs = (*a)*uu;
                f2 = BLAS_ABSSQ( fs );
                h2 = f2 + g2;
            }
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            *c = ( f2*p )*w;
            *s = conj( gs )*( fs*p );
            *a = ( fs*( h2*p ) )*u;
        }
    }

    #undef BLAS_ABSSQ
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTG_HH
