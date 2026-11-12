// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "test.hh"

// -----------------------------------------------------------------------------
template <typename TA, typename TC>
void test_rankk_work( blas::Layout layout, blas::Uplo uplo, double beta )
{
    using blas::Layout, blas::Uplo, blas::Op;
    const TA A[4] = { 1, 2 };
    const double B[4] = { 3, 4 };
    const TC initial[] = {
        1, blas::make_scalar<TC>( 7, 3 ), 41,
        blas::make_scalar<TC>( 11, -5 ), 2, 43
    };

    for (int routine = 0; routine < 4; ++routine) {
        TC C[6];
        std::copy( initial, initial + 6, C );

        // Mixed input/output types select the generic implementations.
        if (routine == 0)
            blas::syrk( layout, uplo, Op::NoTrans, 2, 1,
                        0.0, A, 2, beta, C, 3 );
        else if (routine == 1)
            blas::herk( layout, uplo, Op::NoTrans, 2, 1,
                        0.0, A, 2, beta, C, 3 );
        else if (routine == 2)
            blas::syr2k( layout, uplo, Op::NoTrans, 2, 1,
                         0.0, A, 2, B, 2, beta, C, 3 );
        else
            blas::her2k( layout, uplo, Op::NoTrans, 2, 1,
                         0.0, A, 2, B, 2, beta, C, 3 );

        for (int j = 0; j < 2; ++j) {
            for (int i = 0; i < 2; ++i) {
                int index = (layout == Layout::ColMajor ? i + j*3 : i*3 + j);
                bool selected = (uplo == Uplo::General ||
                                 (uplo == Uplo::Upper && i <= j) ||
                                 (uplo == Uplo::Lower && i >= j));
                TC expected = (selected ? beta*initial[index] : initial[index]);
                require( C[index] == expected );
            }
        }
        require( C[2] == initial[2] );
        require( C[5] == initial[5] );
    }
}

// -----------------------------------------------------------------------------
void test_rankk( Params& params, bool run )
{
    // Fixed-size regression; allow --dim 2 to run each parameter case once.
    (void)params.dim.n();
    blas::Layout layout = params.layout();
    blas::Uplo uplo = params.uplo();
    double beta = params.beta.get<double>();

    if (! run)
        return;

    test_rankk_work< float, double >( layout, uplo, beta );
    test_rankk_work< double, std::complex<double> >( layout, uplo, beta );
    params.okay() = true;
}
