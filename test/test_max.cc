// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"

// -----------------------------------------------------------------------------
void test_max( Params& params, bool run )
{
    int64_t m = params.dim.m();
    if (! run)
        return;

    // the test is mostly in compilation:
    // with std::max, this code won't compile, while with blas::max, it will.
    using blas::max;
    using blas::min;
    //using std::max;
    //using std::min;
    int i = 32;
    int i2 = 64;
    float f = 3.1415;
    double d = 1.2345;

    // -----
    auto max_xi   = max( 1, 2     );
    auto max_xi2  = max( 1, i     );
    auto max_xi3  = max( 1, i, 2  );
    auto max_xi4  = max( 1, i, 2, 4 );
    auto max_xi5  = max( 1, i, 2, 5, i2 );
    auto max_xi64 = max( 1, m     );
    auto max_xd   = max( 1, 0.245 );
    auto max_xd2  = max( 1, d     );
    auto max_xd3  = max( 1, 0.245, d  );
    auto max_xd4  = max( 1, 0.245, f, d );
    auto max_xd5  = max( 1, 0.245, f, d, 3.0 );
    auto max_xs   = max( 1, 1.23f, f );

    // different order
    auto max_yi   = max( 2,     1 );
    auto max_yi2  = max( i,     1 );
    auto max_yi64 = max( m,     1 );
    auto max_yd   = max( 0.245, 1 );
    auto max_yd2  = max( d,     1 );

    // -----
    auto min_xi   = min( 1, 2     );
    auto min_xi2  = min( 1, i     );
    auto min_xi3  = min( 1, i, 2  );
    auto min_xi4  = min( 1, i, 2, 4 );
    auto min_xi5  = min( 1, i, 2, 5, i2 );
    auto min_xi64 = min( 1, m     );
    auto min_xd   = min( 1, 0.245 );
    auto min_xd2  = min( 1, d     );
    auto min_xd3  = min( 1, 0.245, d  );
    auto min_xd4  = min( 1, 0.245, f, d );
    auto min_xd5  = min( 1, 0.245, f, d, 3.0 );
    auto min_xs   = min( 1, 1.23f, f );

    // different order
    auto min_yi   = min( 2,     1 );
    auto min_yi2  = min( i,     1 );
    auto min_yi64 = min( m,     1 );
    auto min_yd   = min( 0.245, 1 );
    auto min_yd2  = min( d,     1 );

    // -----
    // check results
    // use assert, to tell which one failed, vs. single okay flag
    require( max_xi   == 2 );
    require( max_xi2  == i );
    require( max_xi3  == i );
    require( max_xi4  == i );
    require( max_xi5  == i2 );
    require( max_xi64 == m );
    require( max_xd   == 1.0 );
    require( max_xd2  == d );
    require( max_xd3  == d );
    require( max_xd4  == f );
    require( max_xd5  == f );
    require( max_xs   == f );

    require( max_yi   == 2 );
    require( max_yi2  == i );
    require( max_yi64 == m );
    require( max_yd   == 1.0 );
    require( max_yd2  == d );

    require( min_xi   == 1 );
    require( min_xi2  == 1 );
    require( min_xi3  == 1 );
    require( min_xi4  == 1 );
    require( min_xi5  == 1 );
    require( min_xi64 == 1 );
    require( min_xd   == 0.245 );
    require( min_xd2  == 1.0 );
    require( min_xd3  == 0.245 );
    require( min_xd4  == 0.245 );
    require( min_xd5  == 0.245 );
    require( min_xs   == 1.0 );

    require( min_yi   == 1 );
    require( min_yi2  == 1 );
    require( min_yi64 == 1 );
    require( min_yd   == 0.245 );
    require( min_yd2  == 1 );

    // -----
    // check types of results
    // oddly, this can't be done with asserts:
    // error: macro "assert" passed 2 arguments, but takes just 1
    bool okay = true;

    okay = okay && (std::is_same< int,     decltype( max_xi   ) >::value);
    okay = okay && (std::is_same< int,     decltype( max_xi2  ) >::value);
    okay = okay && (std::is_same< int,     decltype( max_xi3  ) >::value);
    okay = okay && (std::is_same< int,     decltype( max_xi4  ) >::value);
    okay = okay && (std::is_same< int,     decltype( max_xi5  ) >::value);
    okay = okay && (std::is_same< int64_t, decltype( max_xi64 ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_xd   ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_xd2  ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_xd3  ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_xd4  ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_xd5  ) >::value);
    okay = okay && (std::is_same< float,   decltype( max_xs   ) >::value);

    okay = okay && (std::is_same< int,     decltype( max_yi   ) >::value);
    okay = okay && (std::is_same< int,     decltype( max_yi2  ) >::value);
    okay = okay && (std::is_same< int64_t, decltype( max_yi64 ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_yd   ) >::value);
    okay = okay && (std::is_same< double,  decltype( max_yd2  ) >::value);

    okay = okay && (std::is_same< int,     decltype( min_xi   ) >::value);
    okay = okay && (std::is_same< int,     decltype( min_xi2  ) >::value);
    okay = okay && (std::is_same< int64_t, decltype( min_xi64 ) >::value);
    okay = okay && (std::is_same< double,  decltype( min_xd   ) >::value);
    okay = okay && (std::is_same< double,  decltype( min_xd2  ) >::value);

    okay = okay && (std::is_same< int,     decltype( min_yi   ) >::value);
    okay = okay && (std::is_same< int,     decltype( min_yi2  ) >::value);
    okay = okay && (std::is_same< int64_t, decltype( min_yi64 ) >::value);
    okay = okay && (std::is_same< double,  decltype( min_yd   ) >::value);
    okay = okay && (std::is_same< double,  decltype( min_yd2  ) >::value);

    params.okay() = okay;
}
