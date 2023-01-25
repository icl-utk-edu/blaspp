// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// blas01_util.cc
// BLAS++ utilities: blas::real_type, blas::is_complex, blas::conj
#include <blas.hh>

#include "util.hh"

//------------------------------------------------------------------------------
template <typename scalar_type>
void test_util( scalar_type alpha )
{
    print_func();

    //--------------------
    // demo blas::real_type
    int64_t n=100;
    std::vector<scalar_type> x( n, 1.0 );

    using real_type = blas::real_type< scalar_type >;
    real_type norm = blas::nrm2( n, x.data(), 1 );
    printf( "norm  %7.4f\n", norm );

    //--------------------
    // demo blas::conj
    scalar_type beta;

    // std::conj fails if alpha is real:
    // error: cannot convert 'std::complex<double>' to 'double' in assignment
    //beta = std::conj( alpha );

    // blas::conj works. Need `using`!
    using blas::conj;
    beta = conj( alpha );

    //--------------------
    // demo blas::is_complex
    using std::real;
    using std::imag;
    if (blas::is_complex<scalar_type>::value) {
        printf( "alpha %7.4f + %7.4fi\n", real(alpha), imag(alpha) );
        printf( "beta  %7.4f + %7.4fi\n", real(beta),  imag(beta)  );
    }
    else {
        printf( "alpha %7.4f\n", real(alpha) );
        printf( "beta  %7.4f\n", real(beta)  );
    }
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    test_util(  float(1.234) );
    //test_util( double(2.468) );
    test_util( std::complex< float>( 3.1415, 0.5678 ) );
    //test_util( std::complex<double>( 6.2830, 1.1356 ) );
}
