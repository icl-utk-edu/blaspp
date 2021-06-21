// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
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
    TA *a,
    TB *b,
    blas::real_type<TA, TB>   *c,
    blas::scalar_type<TA, TB> *s )
{
    typedef real_type<TA, TB> real_t;
    typedef scalar_type<TA, TB> scalar_t;

    #define ABSSQ(t_) real(t_)*real(t_) + imag(t_)*imag(t_)

    // Constants
    const real_t r_one = 1;
    const real_t r_zero = 0;
    const scalar_t zero = 0;
    const TA zero_ta = 0;
    const TB zero_tb = 0;

    // Scaling constants
    const real_t safmin = safe_min<real_t>();
    const real_t safmax = safe_max<real_t>();
    const real_t rtmin = root_min<real_t>();
    const real_t rtmax = root_max<real_t>();

    // Conventions
    const TA& f = *a;
    const TB& g = *b;
    TA& r = *a;

    // quick return
    if (g == zero_tb) {
        *c = r_one;
        *s = zero;
        return;
    }

    if (f == zero_ta) {
        *c = r_zero;
        real_t g1 = max( abs(real(g)), abs(imag(g)) );
        if (g1 > rtmin && g1 < rtmax) {
            // Use unscaled algorithm
            real_t g2 = ABSSQ( g );
            real_t d = sqrt( g2 );
            *s = conj( g ) / d;
            r = d;
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = g*uu;
            real_t g2 = ABSSQ( gs );
            real_t d = sqrt( g2 );
            *s = conj( gs ) / d;
            r = d*u;
        }
    }
    else {
        real_t f1 = max( abs(real(f)), abs(imag(f)) );
        real_t g1 = max( abs(real(g)), abs(imag(g)) );
        if ( f1 > rtmin && f1 < rtmax &&
             g1 > rtmin && g1 < rtmax ) {
            // Use unscaled algorithm
            real_t f2 = ABSSQ( f );
            real_t g2 = ABSSQ( g );
            real_t h2 = f2 + g2;
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            *c = f2*p;
            *s = conj( g )*( f*p );
            r = f*( h2*p );
        }
        else {
            // Use scaled algorithm
            real_t u = min( safmax, max( safmin, f1, g1 ) );
            real_t uu = r_one / u;
            scalar_t gs = g*uu;
            real_t g2 = ABSSQ( gs );
            real_t f2, h2, w;
            scalar_t fs;
            if (f1*uu < rtmin) {
                // f is not well-scaled when scaled by g1.
                real_t v = min( safmax, max( safmin, f1 ) );
                real_t vv = r_one / v;
                w = v * uu;
                fs = f*vv;
                f2 = ABSSQ( fs );
                h2 = f2*w*w + g2;
            }
            else {
                // Otherwise use the same scaling for f and g.
                w = r_one;
                fs = f*uu;
                f2 = ABSSQ( fs );
                h2 = f2 + g2;
            }
            real_t d = ( f2 > rtmin && h2 < rtmax )
                       ? sqrt( f2*h2 )
                       : sqrt( f2 )*sqrt( h2 );
            real_t p = r_one / d;
            *c = ( f2*p )*w;
            *s = conj( gs )*( fs*p );
            r = ( fs*( h2*p ) )*u;
        }
    }

    #undef ABSSQ
}

}  // namespace blas

#endif        //  #ifndef BLAS_ROTG_HH
