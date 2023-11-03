// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"

// -----------------------------------------------------------------------------
// traits class maps data type to real_t and scalar_t.

// for float, double:
// norm and scalar are float, double, respectively.
template <typename T>
class traits
{
public:
    typedef T real_t;
    typedef T scalar_t;
};

// for std::complex<float>, std::complex<double>:
// norm   is float, double, respectively;
// scalar is std::complex<float>, std::complex<double>, respectively.
template <typename T>
class traits< std::complex<T> >
{
public:
    typedef T real_t;
    typedef std::complex<T> scalar_t;
};

// -----------------------------------------------------------------------------
// Example function to test.
// In this case, it sorts the vector x, using C++ std::sort.
//
template <typename scalar_t>
void my_sort( std::vector<scalar_t>& x )
{
    std::sort( x.begin(), x.end() );
}

// -----------------------------------------------------------------------------
// Comparison function for qsort. Returns negative, zero, or positive
// if x is less than, equal to, or greater than y, respectively.
//
template <typename T>
int compare( const void* x, const void* y )
{
    T x_ = ((T*)x)[0];
    T y_ = ((T*)y)[0];
    return (x_ < y_ ? -1 : x_ == y_ ? 0 : 1);
}

// -----------------------------------------------------------------------------
// Example function to test -- reference implementation.
// In this case, it sorts the vector x, using C's qsort.
//
template <typename T>
void ref_sort( std::vector<T>& x )
{
    qsort( &x[0], x.size(), sizeof(T), compare<T> );
}

// -----------------------------------------------------------------------------
// Print vector.
//
template <typename T>
void print( const char* label, std::vector<T>& x )
{
    printf( "%s = [\n", label );
    for (auto x_i: x) {
        printf( "  %9.6f\n", x_i );
    }
    printf( "];\n" );
}

// -----------------------------------------------------------------------------
template <typename scalar_t>
void test_sort_work( Params &params, bool run )
{
    using llong = long long;
    using testsweeper::get_wtime;
    typedef typename traits<scalar_t>::real_t real_t;

    // get & mark input and non-standard output values
    int64_t nb = params.nb();
    int64_t m = params.dim.m();
    int64_t n = params.dim.n();
    int64_t k = params.dim.k();
    int64_t cache = params.cache();
    int verbose = params.verbose();
    bool check = (params.check() == 'y');
    bool ref = (params.ref() == 'y') || check;  // check requires ref
    scalar_t alpha = params.alpha.get<scalar_t>();
    real_t beta = params.beta();
    (void) nb;  // Mark as unused.

    // mark non-standard output values
    params.ref_time();
    params.ref_gflops();

    // adjust header to msec
    params.time.name( "SLATE\ntime (ms)" );
    params.gflops.name( "gflop/s" );
    params.ref_time.name( "LAPACK Reference\ntime (ms)" );
    params.ref_gflops.name( "LAPACK\nreference gflop/s" );

    assert( params.time.width()       == 11 );  // default width
    assert( params.gflops.width()     == 11 );  // default width
    assert( params.ref_time.width()   == 16 );  // LAPACK Reference  (1st line)
    assert( params.ref_gflops.width() == 17 );  // reference gflop/s (2nd line)

    if (! run)
        return;

    // ----------
    // setup
    int64_t imax = 100000;
    size_t len = std::min( m, imax ) + std::min( n, imax ) + std::min( k, imax );
    std::vector<real_t> x( len );
    for (auto& x_i: x) {
        x_i = rand() / double(RAND_MAX) + std::abs( alpha ) + beta;
    }
    std::vector<real_t> x_ref = x;  // copy

    if (verbose >= 2) {
        print( "x_in", x );
    }

    double time;
    double gflop = len * log2( len ) * 1e-9;

    // run test
    testsweeper::flush_cache( cache );
    time = get_wtime();
    my_sort( x );
    time = get_wtime() - time;
    params.time()   = time * 1000;  // msec
    params.gflops() = gflop / time;

    if (verbose >= 2) {
        print( "x_out", x );
    }

    if (ref) {
        // run reference
        testsweeper::flush_cache( cache );
        time = get_wtime();
        ref_sort( x_ref );  // reference implementation
        time = get_wtime() - time;
        params.ref_time()   = time * 1000;  // msec
        params.ref_gflops() = gflop / time;

        if (verbose >= 2) {
            print( "x_ref", x_ref );
        }
    }

    // check error
    if (check) {
        real_t error = 0;
        error = 1.23456e-17 * n;  // placeholder; fails for n >= 900
        for (size_t i = 0; i < len; ++i) {
            error += std::abs( x[i] - x_ref[i] );
        }
        real_t eps = std::numeric_limits< real_t >::epsilon();
        real_t tol = params.tol() * eps;
        params.error() = error;
        params.okay()  = (error < tol);
    }
}

// -----------------------------------------------------------------------------
void test_sort( Params &params, bool run )
{
    switch (params.datatype()) {
        case testsweeper::DataType::Single:
            test_sort_work< float >( params, run );
            break;

        case testsweeper::DataType::Double:
            test_sort_work< double >( params, run );
            break;

        case testsweeper::DataType::SingleComplex:
            test_sort_work< std::complex<float> >( params, run );
            break;

        case testsweeper::DataType::DoubleComplex:
            test_sort_work< std::complex<double> >( params, run );
            break;

        default:
            throw std::runtime_error( "unknown datatype" );
            break;
    }
}

// -----------------------------------------------------------------------------
void test_bar( Params &params, bool run )
{
    test_sort( params, run );
}

// -----------------------------------------------------------------------------
void test_baz( Params &params, bool run )
{
    test_sort( params, run );
}
