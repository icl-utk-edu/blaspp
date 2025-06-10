// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef TEST_HH
#define TEST_HH

#include "testsweeper.hh"
#include "blas.hh"

//------------------------------------------------------------------------------
// For printf, int64_t could be long (%ld), which is >= 32 bits,
// or long long (%lld), guaranteed >= 64 bits.
// Cast to llong to ensure printing 64 bits.
using llong = long long;

//------------------------------------------------------------------------------
namespace blas {

enum class Format : char
{
    LAPACK   = 'L',
    Tile     = 'T',
};

extern const char* Format_help;

inline char to_char( Format format )
{
    return char(format);
}

inline const char* to_c_string( Format value )
{
    switch (value) {
        case Format::LAPACK: return "lapack";
        case Format::Tile:   return "tile";
    }
    return "?";
}

inline std::string to_string( Format value )
{
    return to_c_string( value );
}

inline void from_string( std::string const& str, Format* val )
{
    std::string str_ = str;
    std::transform( str_.begin(), str_.end(), str_.begin(), ::tolower );
    if (str_ == "l" || str_ == "lapack")
        *val = Format::LAPACK;
    else if (str_ == "t" || str_ == "tile")
        *val = Format::Tile;
    else
        throw blas::Error( "unknown Format: " + str );
}

}  // namespace blas


//------------------------------------------------------------------------------
class Params: public testsweeper::ParamsBase {
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    //----- test framework parameters
    testsweeper::ParamChar   check;
    testsweeper::ParamChar   ref;
    testsweeper::ParamChar   papi;
    testsweeper::ParamDouble tol;
    testsweeper::ParamInt    repeat;
    testsweeper::ParamInt    verbose;
    testsweeper::ParamInt    cache;
    std::string              routine;

    //----- routine parameters, enums
    testsweeper::ParamEnum< testsweeper::DataType > datatype;

    // BLAS & LAPACK options
    testsweeper::ParamEnum< blas::Layout >          layout;
    testsweeper::ParamEnum< blas::Format >          format;
    testsweeper::ParamEnum< blas::Side >            side;
    testsweeper::ParamEnum< blas::Uplo >            uplo;
    testsweeper::ParamEnum< blas::Op >              trans;
    testsweeper::ParamEnum< blas::Op >              transA;
    testsweeper::ParamEnum< blas::Op >              transB;
    testsweeper::ParamEnum< blas::Diag >            diag;
    testsweeper::ParamChar                          pointer_mode;

    //----- routine parameters, numeric
    testsweeper::ParamInt3    dim;  // m, n, k
    testsweeper::ParamComplex alpha;
    testsweeper::ParamComplex beta;
    testsweeper::ParamInt     incx;
    testsweeper::ParamInt     incy;
    testsweeper::ParamInt     align;
    testsweeper::ParamInt     batch;
    testsweeper::ParamInt     device;

    //----- output parameters
    testsweeper::ParamScientific error;
    testsweeper::ParamScientific error2;
    testsweeper::ParamScientific error3;

    testsweeper::ParamDouble     time;
    testsweeper::ParamDouble     gflops;
    testsweeper::ParamDouble     gbytes;

    testsweeper::ParamDouble     time2;
    testsweeper::ParamDouble     gflops2;
    testsweeper::ParamDouble     gbytes2;

    testsweeper::ParamDouble     time3;
    testsweeper::ParamDouble     gflops3;
    testsweeper::ParamDouble     gbytes3;

    testsweeper::ParamDouble     time4;
    testsweeper::ParamDouble     gflops4;
    testsweeper::ParamDouble     gbytes4;

    testsweeper::ParamDouble     ref_time;
    testsweeper::ParamDouble     ref_gflops;
    testsweeper::ParamDouble     ref_gbytes;

    testsweeper::ParamOkay       okay;
    testsweeper::ParamString     msg;
};

//------------------------------------------------------------------------------
template <typename T>
inline T roundup( T x, T y )
{
    return T( (x + y - 1) / y ) * y;
}

//------------------------------------------------------------------------------
#ifndef assert_throw
    #if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))
        #define assert_throw( expr, exception_type ) \
            ((void)0)
    #else
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
    #endif
#endif

//------------------------------------------------------------------------------
// Like assert(), but throws error and is not disabled by NDEBUG.
inline
void require_( bool cond, const char* condstr, const char* file, int line )
{
    if (! cond) {
        throw blas::Error( std::string(condstr) + " failed at "
                           + file + ":" + std::to_string(line) );
    }
}

#define require( cond ) require_( (cond), #cond, __FILE__, __LINE__ )

//------------------------------------------------------------------------------
/// Synchronize the GPU queue, then return the current time in seconds.
inline
double sync_get_wtime( blas::Queue& queue )
{
    queue.sync();
    return testsweeper::get_wtime();
}

//------------------------------------------------------------------------------
// Level 1 BLAS
void test_asum  ( Params& params, bool run );
void test_axpy  ( Params& params, bool run );
void test_copy  ( Params& params, bool run );
void test_dot   ( Params& params, bool run );
void test_iamax ( Params& params, bool run );
void test_nrm2  ( Params& params, bool run );
void test_rot   ( Params& params, bool run );
void test_rotg  ( Params& params, bool run );
void test_rotm  ( Params& params, bool run );
void test_rotmg ( Params& params, bool run );
void test_scal  ( Params& params, bool run );
void test_swap  ( Params& params, bool run );

//------------------------------------------------------------------------------
// Level 2 BLAS
void test_gemv  ( Params& params, bool run );
void test_ger   ( Params& params, bool run );
void test_hemv  ( Params& params, bool run );
void test_her   ( Params& params, bool run );
void test_her2  ( Params& params, bool run );
void test_symv  ( Params& params, bool run );
void test_syr   ( Params& params, bool run );
void test_syr2  ( Params& params, bool run );
void test_trmv  ( Params& params, bool run );
void test_trsv  ( Params& params, bool run );

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
// Level 1 GPU BLAS
void test_asum_device  ( Params& params, bool run );
void test_axpy_device  ( Params& params, bool run );
void test_dot_device   ( Params& params, bool run );
void test_iamax_device ( Params& params, bool run );
void test_nrm2_device  ( Params& params, bool run );
void test_rot_device   ( Params& params, bool run );
void test_rotg_device  ( Params& params, bool run );
void test_rotm_device  ( Params& params, bool run );
void test_rotmg_device ( Params& params, bool run );
void test_scal_device  ( Params& params, bool run );
void test_swap_device  ( Params& params, bool run );
void test_copy_device  ( Params& params, bool run );

//------------------------------------------------------------------------------
// Level 2 GPU BLAS
void test_hemv_device  ( Params& params, bool run );
void test_symv_device  ( Params& params, bool run );

//------------------------------------------------------------------------------
// Level 3 GPU BLAS
void test_gemm_device  ( Params& params, bool run );
void test_hemm_device  ( Params& params, bool run );
void test_her2k_device ( Params& params, bool run );
void test_herk_device  ( Params& params, bool run );
void test_schur_gemm   ( Params& params, bool run );
void test_symm_device  ( Params& params, bool run );
void test_syr2k_device ( Params& params, bool run );
void test_syrk_device  ( Params& params, bool run );
void test_trmm_device  ( Params& params, bool run );
void test_trsm_device  ( Params& params, bool run );

void test_batch_gemm_device  ( Params& params, bool run );
void test_batch_hemm_device  ( Params& params, bool run );
void test_batch_her2k_device ( Params& params, bool run );
void test_batch_herk_device  ( Params& params, bool run );
void test_batch_symm_device  ( Params& params, bool run );
void test_batch_syr2k_device ( Params& params, bool run );
void test_batch_syrk_device  ( Params& params, bool run );
void test_batch_trmm_device  ( Params& params, bool run );
void test_batch_trsm_device  ( Params& params, bool run );

//------------------------------------------------------------------------------
// auxiliary
void test_error ( Params& params, bool run );
void test_max   ( Params& params, bool run );
void test_util  ( Params& params, bool run );
void test_memcpy( Params& params, bool run );
void test_memcpy_2d( Params& params, bool run );

#endif        //  #ifndef TEST_HH
