// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TEST_HH
#define TEST_HH

#include "testsweeper.hh"
#include "blas.hh"

// -----------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   ref;
    //testsweeper::ParamDouble tol;  // stricter bounds don't need arbitrary tol
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    cache;

    // ----- routine parameters
    testsweeper::ParamEnum< testsweeper::DataType > datatype;
    testsweeper::ParamEnum< blas::Layout >      layout;
    testsweeper::ParamEnum< blas::Side >        side;
    testsweeper::ParamEnum< blas::Uplo >        uplo;
    testsweeper::ParamEnum< blas::Op >          trans;
    testsweeper::ParamEnum< blas::Op >          transA;
    testsweeper::ParamEnum< blas::Op >          transB;
    testsweeper::ParamEnum< blas::Diag >        diag;

    testsweeper::ParamInt3   dim;
    testsweeper::ParamDouble alpha;
    testsweeper::ParamDouble beta;
    testsweeper::ParamInt    incx;
    testsweeper::ParamInt    incy;
    testsweeper::ParamInt    align;
    testsweeper::ParamInt    batch;
    testsweeper::ParamInt    device;

    // ----- output parameters
    testsweeper::ParamScientific error;
    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;
    testsweeper::ParamDouble     gbytes;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamDouble     ref_gbytes;

    testsweeper::ParamOkay       okay;
};

// -----------------------------------------------------------------------------
template< typename T >
inline T roundup( T x, T y )
{
    return T( (x + y - 1) / y ) * y;
}

// -----------------------------------------------------------------------------
#define assert_throw( expr, exception_type ) \
    try { \
        expr; \
        fprintf( stderr, "Error: didn't throw expected exception at %s:%d\n", \
                 __FILE__, __LINE__ ); \
        throw std::exception(); \
    } \
    catch (exception_type& err) { \
        if (verbose >= 3) { \
            printf( "Caught expected exception: %s\n", err.what() ); \
        } \
    }

// -----------------------------------------------------------------------------
// Level 1 BLAS
void test_asum  ( Params& params, bool run );
void test_axpy  ( Params& params, bool run );
void test_copy  ( Params& params, bool run );
void test_dot   ( Params& params, bool run );
void test_dotu  ( Params& params, bool run );
void test_iamax ( Params& params, bool run );
void test_nrm2  ( Params& params, bool run );
void test_rot   ( Params& params, bool run );
void test_rotg  ( Params& params, bool run );
void test_rotm  ( Params& params, bool run );
void test_rotmg ( Params& params, bool run );
void test_scal  ( Params& params, bool run );
void test_swap  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 2 BLAS
void test_gemv  ( Params& params, bool run );
void test_ger   ( Params& params, bool run );
void test_geru  ( Params& params, bool run );
void test_hemv  ( Params& params, bool run );
void test_her   ( Params& params, bool run );
void test_her2  ( Params& params, bool run );
void test_symv  ( Params& params, bool run );
void test_syr   ( Params& params, bool run );
void test_syr2  ( Params& params, bool run );
void test_trmv  ( Params& params, bool run );
void test_trsv  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 BLAS
void test_gemm  ( Params& params, bool run );
void test_hemm  ( Params& params, bool run );
void test_her2k ( Params& params, bool run );
void test_herk  ( Params& params, bool run );
void test_symm  ( Params& params, bool run );
void test_syr2k ( Params& params, bool run );
void test_syrk  ( Params& params, bool run );
void test_trmm  ( Params& params, bool run );
void test_trsm  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 Batch BLAS
void test_batch_gemm  ( Params& params, bool run );
void test_batch_hemm  ( Params& params, bool run );
void test_batch_her2k ( Params& params, bool run );
void test_batch_herk  ( Params& params, bool run );
void test_batch_symm  ( Params& params, bool run );
void test_batch_syr2k ( Params& params, bool run );
void test_batch_syrk  ( Params& params, bool run );
void test_batch_trmm  ( Params& params, bool run );
void test_batch_trsm  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 GPU BLAS
#ifdef BLASPP_WITH_CUBLAS
void test_gemm_device  ( Params& params, bool run );
void test_trsm_device  ( Params& params, bool run );
void test_trmm_device  ( Params& params, bool run );
void test_hemm_device  ( Params& params, bool run );
void test_symm_device  ( Params& params, bool run );
void test_herk_device  ( Params& params, bool run );
void test_syrk_device  ( Params& params, bool run );
void test_her2k_device  ( Params& params, bool run );
void test_syr2k_device  ( Params& params, bool run );

void test_batch_gemm_device( Params& params, bool run );
void test_batch_trsm_device( Params& params, bool run );
void test_batch_trmm_device( Params& params, bool run );
void test_batch_hemm_device( Params& params, bool run );
void test_batch_symm_device( Params& params, bool run );
void test_batch_herk_device( Params& params, bool run );
void test_batch_syrk_device( Params& params, bool run );
void test_batch_her2k_device( Params& params, bool run );
void test_batch_syr2k_device( Params& params, bool run );
#endif

// -----------------------------------------------------------------------------
// auxiliary
void test_error ( Params& params, bool run );
void test_max   ( Params& params, bool run );
void test_util  ( Params& params, bool run );

#endif  //  #ifndef TEST_HH
