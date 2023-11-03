// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef EXAMPLE_HH
#define EXAMPLE_HH

#include "testsweeper.hh"

// -----------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   ref;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    cache;

    // ----- routine parameters
    testsweeper::ParamEnum< testsweeper::DataType > datatype_old;
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamInt    nb;
    testsweeper::ParamInt3   dim;
    testsweeper::ParamInt3   grid;
    testsweeper::ParamComplex alpha;
    testsweeper::ParamDouble beta;

    // ----- output parameters
    testsweeper::ParamScientific error;
    testsweeper::ParamScientific ortho;
    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;

    testsweeper::ParamScientific ref_error;
    testsweeper::ParamScientific ref_ortho;
    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;

    testsweeper::ParamOkay       okay;
};

// -----------------------------------------------------------------------------
// Level 1
void test_sort( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 2
void test_bar( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3
void test_baz( Params& params, bool run );

#endif  //  #ifndef EXAMPLE_HH
