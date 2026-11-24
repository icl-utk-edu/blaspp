// Copyright (c) 2017-2026, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef PRINT_HH
#define PRINT_HH

#include <assert.h>
#include <complex>
#include <type_traits>

//========================================
// Copied from slate/src/print.cc, with minor modifications.

//------------------------------------------------------------------------------
/// @return 10^y for 0 <= y <= 20.
static inline double pow10( int y )
{
    static double values[] = {
        1, 10, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12,
        1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19, 1e20 };
    assert( 0 <= y && y <= 20 );
    return values[ y ];
}

//------------------------------------------------------------------------------
/// Print real value to a buffer buf of length buf_len.
/// For w = width and p = precision:
/// - integers are printed with %v.0f where v = w-p
/// - small values < 0.01 or large values > threshold are printed with %w.pg
/// - modest values are printed with %w.pf.
/// To ensure data fits, set threshold = 10^(w - p - 2) and w >= p + 6.
///
/// @return actual bytes written, not including \0 terminator. This
/// differs from snprintf, which returns number of bytes required.
///
template <typename real_t>
int snprintf_value(
    char* buf, int buf_len,
    int width, int precision,
    real_t value,
    std::enable_if_t< std::is_floating_point_v< real_t > >* z=nullptr )
{
    real_t abs_val = std::abs( value );
    real_t threshold = pow10( width - precision - 2 );

    int len;
    if (value == int64_t( value )) {
        // exactly integer, print without digits after decimal point
        len = snprintf( buf, buf_len,
                        " %#*.0f%*s", width - precision, value,
                        precision, "" );
    }
    else if (abs_val < 0.01 || abs_val >= threshold) {
        // small or large value: print with %g
        len = snprintf( buf, buf_len,
                        " %#*.*g", width, precision, value );
    }
    else {
        // between 1 and threshold = 10^(w-p-2): %f will fit in width.
        len = snprintf( buf, buf_len,
                        " %#*.*f", width, precision, value );
    }
    // return bytes actually written.
    return std::min( len, buf_len - 1 );
}

//------------------------------------------------------------------------------
/// Overload to print integer values.
template <typename integer_t>
int snprintf_value(
    char* buf, int buf_len,
    int width, int precision,
    integer_t value,
    std::enable_if_t< std::is_integral_v< integer_t > >* z=nullptr)
{
    int len = snprintf( buf, buf_len, " %*lld", width, llong( value ) );
    // return bytes actually written.
    return std::min( len, buf_len - 1 );
}

//------------------------------------------------------------------------------
/// Overload to print complex values as " <real> + <imag>i".
template <typename real_t>
int snprintf_value(
    char* buf, int buf_len,
    int width, int precision,
    std::complex<real_t> value)
{
    // " real"
    real_t re = std::real( value );
    int len = snprintf_value( buf, buf_len, width, precision, re );
    buf     += len;
    buf_len -= len;

    real_t im = std::imag( value );
    if (im == 0) {
        // blank padding
        int l = snprintf( buf, buf_len, "   %*s ", width, "" );
        l = std::min( l, buf_len - 1 );
        len += l;
    }
    else {
        // " + imagi"
        int l = snprintf( buf, buf_len, " +" );
        l = std::min( l, buf_len - 1 );
        len     += l;
        buf     += l;
        buf_len -= l;

        l = snprintf_value( buf, buf_len, width, precision, im );
        len     += l;
        buf     += l;
        buf_len -= l;

        l = snprintf( buf, buf_len, "i" );
        l = std::min( l, buf_len - 1 );
        len += l;
    }
    return len;
}

// end copy
//========================================

//------------------------------------------------------------------------------
/// Print matrix meta data and values in Matlab format.
/// For verbose = 0, no output.
/// For verbose = 1, prints comment with label and dimensions (m, n, lda).
/// For verbose = 2, prints abbreviated data by printing the first
/// and last edgeitems rows and columns, with ellipsis between.
/// For verbose = 3, prints abbreviated data by printing every nb rows
/// and columns, without ellipsis.
/// For verbose = 4, prints all data.
///
template <typename T>
void print_matrix( const char* label, int64_t m, int64_t n, T* A, int64_t lda,
                   int verbose=2, int precision=4, int edgeitems=4, int nb=32 )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );

    if (verbose == 0)
        return;

    printf( "%% %s %lld-by-%lld matrix, ld %lld\n",
            label, llong(m), llong(n), llong(lda) );

    if (verbose >= 2) {
        int width = precision + 6;
        int step = (verbose == 3 ? nb : 1);
        if (verbose == 3)
            printf( "%% printing every %d-th row and col.\n", nb );

        std::string msg;
        char buf[ 80 ];
        printf( "%s = [\n", label );
        for (int64_t i = 0; i < m; i += step) {
            msg = "";
            for (int64_t j = 0; j < n; j += step) {
                snprintf_value( buf, sizeof(buf), width, precision, A(i, j) );
                msg += buf;
                if (verbose == 2 && j == edgeitems - 1 && j < n - edgeitems - 1) {
                    msg += "  ...";
                    j = n - edgeitems - 1;
                }
            }
            printf( "%s\n", msg.c_str() );
            if (verbose == 2 && i == edgeitems - 1 && i < m - edgeitems - 1) {
                printf( "     %% ...\n" );
                i = m - edgeitems - 1;
            }
        }
        printf( "];\n" );
    }

    #undef A
}

//------------------------------------------------------------------------------
/// Prints vector meta data and values in Matlab format.
/// @see print_matrix for details.
///
template <typename T>
void print_vector( const char* label, int64_t n, T *x, int64_t incx,
                   int verbose=2, int precision=4, int edgeitems=4, int nb=32 )
{
    assert( n >= 0 );
    assert( incx != 0 );

    if (verbose == 0)
        return;

    printf( "%% %s %lld vector\n", label, llong(n) );

    if (verbose >= 2) {
        int width = precision + 6;
        int step = (verbose == 3 ? nb : 1);
        if (verbose == 3)
            printf( "%% printing every %d-th entry.\n", nb );

        std::string msg;
        char buf[ 80 ];
        printf( "%s = [", label );
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; i += step) {
            snprintf_value( buf, sizeof(buf), width, precision, x[ix] );
            msg += buf;
            ix += incx;
            if (verbose == 2 && i == edgeitems - 1 && i < n - edgeitems - 1) {
                msg += "  ...";
                i = n - edgeitems - 1;
            }
        }
        printf( "%s ]';\n", msg.c_str() );
    }
}

#endif  // PRINT_HH
