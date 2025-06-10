// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <complex>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "blas/counter.hh"

#ifdef BLAS_HAVE_PAPI
    #include "papi.h"
#endif

#include "test.hh"

// Headers for versions.
#ifdef BLAS_HAVE_MKL
    #include <mkl.h>
#endif
#ifdef BLAS_HAVE_OPENBLAS
    #include <cblas.h>
#endif
#ifdef BLAS_HAVE_CUBLAS
    #include <cuda_runtime.h>
#endif
#ifdef BLAS_HAVE_ROCBLAS
    #include <rocm-core/rocm_version.h>
#endif

// -----------------------------------------------------------------------------
using testsweeper::ParamType;
using testsweeper::DataType;
using testsweeper::DataType_help;

using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;

using blas::Layout, blas::Layout_help;
using blas::Side,   blas::Side_help;
using blas::Uplo,   blas::Uplo_help;
using blas::Op,     blas::Op_help;
using blas::Diag,   blas::Diag_help;
using blas::Format;

const char* Format_help = "one of: L or LAPACK; T or Tile";

const ParamType PT_Value = ParamType::Value;
const ParamType PT_List  = ParamType::List;
const ParamType PT_Out   = ParamType::Output;

const double no_data = testsweeper::no_data_flag;
const char*  pi_rt2i = "3.141592653589793 + 1.414213562373095i";
const char*  e_rt3i  = "2.718281828459045 + 1.732050807568877i";
const double pi      = 3.141592653589793;
const double e       = 2.718281828459045;

// -----------------------------------------------------------------------------
// each section must have a corresponding entry in section_names
enum Section {
    newline = 0,  // zero flag forces newline
    blas1,
    blas2,
    blas3,
    device_blas1,
    device_blas2,
    device_blas3,
    aux,
    num_sections,  // last
};

const char* section_names[] = {
   "",  // none
   "Level 1 BLAS",
   "Level 2 BLAS",
   "Level 3 BLAS",
   "Level 1 BLAS (Device)",
   "Level 2 BLAS (Device)",
   "Level 3 BLAS (Device)",
   "auxiliary",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< testsweeper::routines_t > routines = {
    // Level 1 BLAS
    { "asum",   test_asum,   Section::blas1   },
    { "axpy",   test_axpy,   Section::blas1   },
    { "copy",   test_copy,   Section::blas1   },
    { "dot",    test_dot,    Section::blas1   },
    { "dotu",   test_dot,    Section::blas1   },
    { "iamax",  test_iamax,  Section::blas1   },
    { "nrm2",   test_nrm2,   Section::blas1   },
    { "rot",    test_rot,    Section::blas1   },
    { "rotg",   test_rotg,   Section::blas1   },
    { "rotm",   test_rotm,   Section::blas1   },
    { "rotmg",  test_rotmg,  Section::blas1   },
    { "scal",   test_scal,   Section::blas1   },
    { "swap",   test_swap,   Section::blas1   },
    { "",       nullptr,     Section::newline },

    // Level 2 BLAS
    { "gemv",   test_gemv,   Section::blas2   },
    { "ger",    test_ger,    Section::blas2   },
    { "geru",   test_ger,    Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "hemv",   test_hemv,   Section::blas2   },
    { "her",    test_her,    Section::blas2   },
    { "her2",   test_her2,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "symv",   test_symv,   Section::blas2   },
    { "syr",    test_syr,    Section::blas2   },
    { "syr2",   test_syr2,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    { "trmv",   test_trmv,   Section::blas2   },
    { "trsv",   test_trsv,   Section::blas2   },
    { "",       nullptr,     Section::newline },

    // Level 3 BLAS
    { "gemm",   test_gemm,   Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "hemm",   test_hemm,   Section::blas3   },
    { "herk",   test_herk,   Section::blas3   },
    { "her2k",  test_her2k,  Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "symm",   test_symm,   Section::blas3   },
    { "syrk",   test_syrk,   Section::blas3   },
    { "syr2k",  test_syr2k,  Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "trmm",   test_trmm,   Section::blas3   },
    { "trsm",   test_trsm,   Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "batch-gemm",   test_batch_gemm,   Section::blas3   },
    { "",             nullptr,           Section::newline },

    { "batch-hemm",   test_batch_hemm,   Section::blas3   },
    { "batch-herk",   test_batch_herk,   Section::blas3   },
    { "batch-her2k",  test_batch_her2k,  Section::blas3   },
    { "",             nullptr,           Section::newline },

    { "batch-symm",   test_batch_symm,   Section::blas3   },
    { "batch-syrk",   test_batch_syrk,   Section::blas3   },
    { "batch-syr2k",  test_batch_syr2k,  Section::blas3   },
    { "",              nullptr,          Section::newline },

    { "batch-trmm",   test_batch_trmm,   Section::blas3   },
    { "batch-trsm",   test_batch_trsm,   Section::blas3   },
    { "",              nullptr,          Section::newline },

    // Device Level 1 BLAS
    { "dev-asum",         test_asum_device,         Section::device_blas1   },
    { "dev-axpy",         test_axpy_device,         Section::device_blas1   },
    { "dev-dot",          test_dot_device,          Section::device_blas1   },
    { "dev-dotu",         test_dot_device,          Section::device_blas1   },
    { "dev-iamax",        test_iamax_device,        Section::device_blas1   },
    { "dev-nrm2",         test_nrm2_device,         Section::device_blas1   },
    { "dev-rot",          test_rot_device,          Section::device_blas1   },
    { "dev-rotg",         test_rotg_device,         Section::device_blas1   },
    { "dev-rotm",         test_rotm_device,         Section::device_blas1   },
    { "dev-rotmg",        test_rotmg_device,        Section::device_blas1   },
    { "dev-scal",         test_scal_device,         Section::device_blas1   },
    { "dev-swap",         test_swap_device,         Section::device_blas1   },
    { "dev-copy",         test_copy_device,         Section::device_blas1   },
    { "",                 nullptr,                  Section::newline },

    // Device Level 2 BLAS
    { "dev-symv",         test_symv_device,         Section::device_blas2   },
    { "",                 nullptr,                  Section::newline },

    // Device Level 3 BLAS
    { "dev-gemm",         test_gemm_device,         Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-hemm",         test_hemm_device,         Section::device_blas3   },
    { "dev-herk",         test_herk_device,         Section::device_blas3   },
    { "dev-her2k",        test_her2k_device,        Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-symm",         test_symm_device,         Section::device_blas3   },
    { "dev-syrk",         test_syrk_device,         Section::device_blas3   },
    { "dev-syr2k",        test_syr2k_device,        Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "schur-gemm",       test_schur_gemm,          Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-trmm",         test_trmm_device,         Section::device_blas3   },
    { "dev-trsm",         test_trsm_device,         Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-batch-gemm",   test_batch_gemm_device,   Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-batch-hemm",   test_batch_hemm_device,   Section::device_blas3   },
    { "dev-batch-herk",   test_batch_herk_device,   Section::device_blas3   },
    { "dev-batch-her2k",  test_batch_her2k_device,  Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-batch-symm",   test_batch_symm_device,   Section::device_blas3   },
    { "dev-batch-syrk",   test_batch_syrk_device,   Section::device_blas3   },
    { "dev-batch-syr2k",  test_batch_syr2k_device,  Section::device_blas3   },
    { "",                 nullptr,                  Section::newline },

    { "dev-batch-trmm",   test_batch_trmm_device,   Section::device_blas3   },
    { "dev-batch-trsm",   test_batch_trsm_device,   Section::device_blas3   },
    { "",                 nullptr,                  Section::newline        },

    // auxiliary
    { "error",            test_error,               Section::aux            },
    { "max",              test_max,                 Section::aux            },
    { "util",             test_util,                Section::aux            },
    { "",                 nullptr,                  Section::newline        },

    { "memcpy",           test_memcpy,              Section::aux            },
    { "copy_vector",      test_memcpy,              Section::aux            },
    { "set_vector",       test_memcpy,              Section::aux            },
    { "",                 nullptr,                  Section::newline        },

    { "memcpy_2d",        test_memcpy_2d,           Section::aux            },
    { "copy_matrix",      test_memcpy_2d,           Section::aux            },
    { "set_matrix",       test_memcpy_2d,           Section::aux            },
    { "",                 nullptr,                  Section::newline        },
};

// -----------------------------------------------------------------------------
// Params class
// List of parameters

Params::Params():
    ParamsBase(),

    // w = width
    // p = precision
    //----- test framework parameters
    //          name,         w, type, default, valid, help
    check     ( "check",      0, PT_Value, 'y', "ny", "check the results" ),
    ref       ( "ref",        0, PT_Value, 'n', "ny", "run reference; sometimes check implies ref" ),
    papi      ( "papi",       0, PT_Value, 'n', "ny", "run papi instrumentation" ),

    //          name,         w, p, type, default,  min,  max, help
    tol       ( "tol",        0, 0, PT_Value,  20,    1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",     0,    PT_Value,   1,    1, 1000, "times to repeat each test" ),
    verbose   ( "verbose",    0,    PT_Value,   0,    0,   10, "verbose level" ),
    cache     ( "cache",      0,    PT_Value,  20,    1, 1024, "total cache size, in MiB" ),

    //----- routine parameters, enums
    //          name,         w, type,    default, help
    datatype  ( "type",       4, PT_List, DataType::Double, DataType_help ),

    // BLAS & LAPACK options
    layout    ( "layout",     6, PT_List, Layout::ColMajor, Layout_help ),
    format    ( "format",     6, PT_List, Format::LAPACK, Format_help ),
    side      ( "side",       6, PT_List, Side::Left, Side_help ),
    uplo      ( "uplo",       6, PT_List, Uplo::Lower, Uplo_help ),
    trans     ( "trans",      7, PT_List, Op::NoTrans, Op_help ),
    transA    ( "transA",     7, PT_List, Op::NoTrans, Op_help ),
    transB    ( "transB",     7, PT_List, Op::NoTrans, Op_help ),
    diag      ( "diag",       7, PT_List, Diag::NonUnit, Diag_help ),
    pointer_mode( "ptr",      3, PT_List, 'h', "hd", "one of: h or host; d or device" ),

    //----- routine parameters, numeric
    //          name,         w, p, type,    default,  min,  max, help
    dim       ( "dim",        6,    PT_List,             0, 1e10, "m by n by k dimensions" ),
    alpha     ( "alpha",      3, 1, PT_List, pi_rt2i, -inf,  inf, "scalar alpha" ),
    beta      ( "beta",       3, 1, PT_List,  e_rt3i, -inf,  inf, "scalar beta" ),
    incx      ( "incx",       4,    PT_List,       1, -1e3,  1e3, "stride of x vector" ),
    incy      ( "incy",       4,    PT_List,       1, -1e3,  1e3, "stride of y vector" ),
    align     ( "align",      0,    PT_List,       1,    1, 1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),
    batch     ( "batch",      6,    PT_List,     100,    0,  1e6, "batch size" ),
    device    ( "device",     6,    PT_List,       0,    0,  100, "device id" ),

    //----- output parameters
    // min, max are ignored
    // error:   %8.2e allows 9.99e-99
    // time:    %9.3f allows 99999.999 s = 2.9 days
    // gflops: %12.3f allows 99999999.999 Gflop/s = 100 Pflop/s
    //          name,         w, p, type,   default, min, max, help
    error     ( "error",      8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error2    ( "error2",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),
    error3    ( "error3",     8, 2, PT_Out, no_data, 0, 0, "numerical error" ),

    time      ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "time to solution" ),
    gflops    ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes    ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),

    time2     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    gflops2   ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes2   ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),

    time3     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    gflops3   ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes3   ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),

    time4     ( "time (s)",   9, 3, PT_Out, no_data, 0, 0, "extra timer" ),
    gflops4   ( "gflop/s",   12, 3, PT_Out, no_data, 0, 0, "Gflop/s rate" ),
    gbytes4   ( "gbyte/s",   12, 3, PT_Out, no_data, 0, 0, "Gbyte/s rate" ),

    ref_time  ( "ref time (s)",  9, 3, PT_Out, no_data, 0, 0, "reference time to solution" ),
    ref_gflops( "ref gflop/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gflop/s rate" ),
    ref_gbytes( "ref gbyte/s",  12, 3, PT_Out, no_data, 0, 0, "reference Gbyte/s rate" ),

    // default -1 means "no check"
    //          name,         w, type, default, min, max, help
    okay      ( "status",     6, PT_Out,    -1, 0, 0, "success indicator" ),
    msg       ( "",           1, PT_Out,    "",       "error message" )
{
    // set header different than command line prefix
    pointer_mode.name("ptr", "pointer-mode");

    // mark standard set of output fields as used
    okay();
    error();
    time();

    // mark framework parameters as used, so they will be accepted on the command line
    check();
    ref();
    repeat();
    verbose();
    cache();
    papi();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
void setup_PAPI( int *event_set )
{
    blas::counter::get();
    #ifdef BLAS_HAVE_PAPI
        require( PAPI_library_init( PAPI_VER_CURRENT ) == PAPI_VER_CURRENT );

        require( PAPI_create_eventset( event_set ) == PAPI_OK );

        require( PAPI_add_named_event( *event_set, "sde:::blas::counter" ) == PAPI_OK );
        require( PAPI_add_named_event( *event_set, "sde:::blas::flops" ) == PAPI_OK );
    #endif
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using testsweeper::QuitException;

    // These may or may not be used; mark unused to silence warnings.
    blas_unused( pi_rt2i );
    blas_unused( e_rt3i  );
    blas_unused( pi      );
    blas_unused( e       );

    // check that all sections have names
    require( sizeof(section_names)/sizeof(*section_names) == Section::num_sections );

    int status = 0;
    try {
        int version = blas::blaspp_version();
        printf( "BLAS++ version %d.%02d.%02d, id %s",
                version / 10000, (version % 10000) / 100, version % 100,
                blas::blaspp_id() );

        // CPU BLAS
        // todo: Accelerate, IBM ESSL, Cray LibSci, BLIS, etc. version.
        #ifdef BLAS_HAVE_ACCELERATE
            printf( ", Apple Accelerate" );
        #endif
        #ifdef BLAS_HAVE_ESSL
            printf( ", IBM ESSL" );
        #endif
        #ifdef BLAS_HAVE_MKL
            MKLVersion mkl_version;
            mkl_get_version( &mkl_version );
            printf( ", Intel MKL %d.%d.%d",
                    mkl_version.MajorVersion,
                    mkl_version.MinorVersion,
                    mkl_version.UpdateVersion );
        #endif
        #ifdef BLAS_HAVE_OPENBLAS
            printf( ", %s", OPENBLAS_VERSION );
        #endif

        // GPU BLAS
        #ifdef BLAS_HAVE_CUBLAS
            printf( ", CUDA %d.%d.%d",
                    CUDART_VERSION / 1000,
                    (CUDART_VERSION % 1000) / 100,
                    CUDART_VERSION % 10 );
        #endif
        #ifdef BLAS_HAVE_ROCBLAS
            printf( ", ROCm %d.%d.%d",
                    ROCM_VERSION_MAJOR,
                    ROCM_VERSION_MINOR,
                    ROCM_VERSION_PATCH );
        #endif
        #ifdef BLAS_HAVE_SYCL
            printf( ", SYCL" );
        #endif
        printf( "\n" );

        // print input so running `test [input] > out.txt` documents input
        printf( "input: %s", argv[0] );
        for (int i = 1; i < argc; ++i) {
            // quote arg if necessary
            std::string arg( argv[i] );
            const char* wordchars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-=";
            if (arg.find_first_not_of( wordchars ) != std::string::npos)
                printf( " '%s'", argv[i] );
            else
                printf( " %s", argv[i] );
        }
        printf( "\n" );

        // Usage: test [params] routine
        if (argc < 2
            || strcmp( argv[argc-1], "-h" ) == 0
            || strcmp( argv[argc-1], "--help" ) == 0)
        {
            usage( argc, argv, routines, section_names );
            throw QuitException();
        }

        // find routine to test
        const char* routine = argv[ argc-1 ];
        testsweeper::test_func_ptr test_routine = find_tester( routine, routines );
        if (test_routine == nullptr) {
            usage( argc, argv, routines, section_names );
            throw std::runtime_error(
                std::string("routine ") + routine + " not found" );
        }

        // mark fields that are used (run=false)
        Params params;
        params.routine = routine;
        test_routine( params, false );

        // Parse parameters up to routine name.
        try {
            params.parse( routine, argc-2, argv+1 );
        }
        catch (const std::exception& ex) {
            params.help( routine );
            throw;
        }

        // show align column if it has non-default values
        if (params.align.size() != 1 || params.align() != 1) {
            params.align.width( 5 );
        }

        #ifdef BLAS_HAVE_PAPI
            int event_set = PAPI_NULL;
            long long counter_values[2];

            if (params.papi() == 'y') {
                // initialize papi
                setup_PAPI( &event_set );

                require( PAPI_start( event_set ) == PAPI_OK );
            }
        #endif

        // run tests
        int repeat = params.repeat();
        std::vector<double> times( repeat ), gflops( repeat );
        testsweeper::DataType last = params.datatype();
        params.header();
        do {
            if (params.datatype() != last) {
                last = params.datatype();
                printf( "\n" );
            }
            for (int iter = 0; iter < repeat; ++iter) {
                try {
                    test_routine( params, true );
                }
                catch (const std::exception& ex) {
                    fprintf( stderr, "%s%sError: %s%s\n",
                             ansi_bold, ansi_red, ex.what(), ansi_normal );
                    params.okay() = false;
                }

                // Collect stats.
                times [ iter ] = params.time();
                gflops[ iter ] = params.gflops();

                params.print();
                fflush( stdout );

                #ifdef BLAS_HAVE_PAPI
                    if (params.papi() == 'y') {
                        // stop papi
                        require( PAPI_read( event_set, counter_values ) == PAPI_OK );

                        // print papi instrumentation
                        printf("FLOP count: %lld\n", counter_values[1]);
                    }
                #endif

                status += ! params.okay();
                params.reset_output();
            }
            if (repeat > 1) {
                testsweeper::print_stats( params.time,   times  );
                testsweeper::print_stats( params.gflops, gflops );
                printf( "\n" );
            }
        } while(params.next());

        #ifdef BLAS_HAVE_PAPI
            if (params.papi() == 'y') {
                // stop papi
                require( PAPI_stop( event_set, counter_values ) == PAPI_OK );

                // print papi instrumentation
                blas::counter::print( (cset_list_object_t *)counter_values[0] );
                printf("FLOP count: %lld\n", counter_values[1]);
                printf( "\n" );
            }
        #endif

        if (status) {
            printf( "%d tests FAILED for %s.\n", status, routine );
        }
        else {
            printf( "All tests passed for %s.\n", routine );
        }
    }
    catch (const QuitException& ex) {
        // pass: no error to print
    }
    catch (const std::exception& ex) {
        fprintf( stderr, "\n%s%sError: %s%s\n",
                 ansi_bold, ansi_red, ex.what(), ansi_normal );
        status = -1;
    }

    return status;
}
