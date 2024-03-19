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
using testsweeper::char2datatype;
using testsweeper::datatype2char;
using testsweeper::datatype2str;
using testsweeper::ansi_bold;
using testsweeper::ansi_red;
using testsweeper::ansi_normal;

const double no_data = testsweeper::no_data_flag;

// const ParamType PT_Value  = ParamType::Value; currently unused
// const ParamType PT_List   = ParamType::List; currently unused
const ParamType PT_Output = ParamType::Output;

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
    { "dotu",   test_dotu,   Section::blas1   },
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
    { "geru",   test_geru,   Section::blas2   },
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
    { "dev-axpy",         test_axpy_device,         Section::device_blas1   },
    { "dev-dot",          test_dot_device,          Section::device_blas1   },
    { "dev-dotu",         test_dotu_device,         Section::device_blas1   },
    { "dev-nrm2",         test_nrm2_device,         Section::device_blas1   },
    { "dev-scal",         test_scal_device,         Section::device_blas1   },
    { "dev-swap",         test_swap_device,         Section::device_blas1   },
    { "dev-copy",         test_copy_device,         Section::device_blas1   },
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
    // def = default
    // ----- test framework parameters
    //         name,       w,    type,         default, valid, help
    check     ( "check",   0,    ParamType::Value, 'y', "ny",  "check the results" ),
    ref       ( "ref",     0,    ParamType::Value, 'n', "ny",  "run reference; sometimes check -> ref" ),
    papi      ( "papi",    0,    ParamType::Value, 'n', "ny",  "run papi instrumentation" ),

    //          name,      w, p, type,         default, min,  max, help
    repeat    ( "repeat",  0,    ParamType::Value,   1,   1, 1000, "times to repeat each test" ),
    verbose   ( "verbose", 0,    ParamType::Value,   0,   0,   10, "verbose level" ),
    cache     ( "cache",   0,    ParamType::Value,  20,   1, 1024, "total cache size, in MiB" ),

    // ----- routine parameters
    //          name,      w,    type,            def,                    char2enum,         enum2char,         enum2str,         help
    datatype  ( "type",    4,    ParamType::List, DataType::Double,       char2datatype,     datatype2char,     datatype2str,     "s=single (float), d=double, c=complex-single, z=complex-double" ),
    layout    ( "layout",  6,    ParamType::List, blas::Layout::ColMajor, blas::char2layout, blas::layout2char, blas::layout2str, "layout: r=row major, c=column major" ),
    format    ( "format",  6,    ParamType::List, blas::Format::LAPACK,   blas::char2format, blas::format2char, blas::format2str, "format: l=lapack, t=tile" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0,     1e9, "m by n by k dimensions" ),
    alpha     ( "alpha",   9, 4, ParamType::List,  pi,  -inf,     inf, "scalar alpha" ),
    beta      ( "beta",    9, 4, ParamType::List,   e,  -inf,     inf, "scalar beta" ),
    incx      ( "incx",    4,    ParamType::List,   1, -1000,    1000, "stride of x vector" ),
    incy      ( "incy",    4,    ParamType::List,   1, -1000,    1000, "stride of y vector" ),
    align     ( "align",   0,    ParamType::List,   1,     1,    1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),
    batch     ( "batch",   6,    ParamType::List, 100,     0,     1e6, "batch size" ),
    device    ( "device",  6,    ParamType::List,   0,     0,     100, "device id" ),
    pointer_mode ( "pointer-mode",  3,    ParamType::List, 'h',  "hd",          "h == host, d == device" ),

    // ----- output parameters
    // min, max are ignored
    //          name,            w, p, type,      default, min, max, help
    // error: %8.2e allows 9.99e-99
    error     ( "error",         8, 2, PT_Output, no_data, 0, 0, "numerical error" ),
    error2    ( "error2",        8, 2, PT_Output, no_data, 0, 0, "numerical error 2" ),
    error3    ( "error3",        8, 2, PT_Output, no_data, 0, 0, "numerical error 3" ),

    // time:    %9.3f allows 99999.999 s = 2.9 days (ref headers need %12)
    // gflops: %12.3f allows 99999999.999 Gflop/s = 100 Pflop/s
    time      ( "time (s)",      9, 3, PT_Output, no_data, 0, 0, "time to solution" ),
    gflops    ( "gflop/s",      12, 3, PT_Output, no_data, 0, 0, "Gflop/s rate" ),
    gbytes    ( "gbyte/s",      12, 3, PT_Output, no_data, 0, 0, "Gbyte/s rate" ),

    time2     ( "time (s)",      9, 3, PT_Output, no_data, 0, 0, "time to solution (2)" ),
    gflops2   ( "gflop/s",      12, 3, PT_Output, no_data, 0, 0, "Gflop/s rate (2)" ),
    gbytes2   ( "gbyte/s",      12, 3, PT_Output, no_data, 0, 0, "Gbyte/s rate (2)" ),

    time3     ( "time (s)",      9, 3, PT_Output, no_data, 0, 0, "time to solution (3)" ),
    gflops3   ( "gflop/s",      12, 3, PT_Output, no_data, 0, 0, "Gflop/s rate (3)" ),
    gbytes3   ( "gbyte/s",      12, 3, PT_Output, no_data, 0, 0, "Gbyte/s rate (3)" ),

    time4     ( "time (s)",      9, 3, PT_Output, no_data, 0, 0, "time to solution (4)" ),
    gflops4   ( "gflop/s",      12, 3, PT_Output, no_data, 0, 0, "Gflop/s rate (4)" ),
    gbytes4   ( "gbyte/s",      12, 3, PT_Output, no_data, 0, 0, "Gbyte/s rate (4)" ),

    ref_time  ( "ref time (s)", 12, 3, PT_Output, no_data, 0, 0, "reference time to solution" ),
    ref_gflops( "ref gflop/s",  12, 3, PT_Output, no_data, 0, 0, "reference Gflop/s rate" ),
    ref_gbytes( "ref gbyte/s",  12, 3, PT_Output, no_data, 0, 0, "reference Gbyte/s rate" ),

    // default -1 means "no check"
    okay      ( "status",              6,    ParamType::Output,  -1,   0,   0, "success indicator" ),
    msg       ( "",       1, ParamType::Output,  "",           "error message" )
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
    #endif
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    using testsweeper::QuitException;

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
            long long counter_values[1];

            if (params.papi() == 'y') {
                // initialize papi
                setup_PAPI( &event_set );

                require( PAPI_start( event_set ) == PAPI_OK );
            }
        #endif

        // run tests
        int repeat = params.repeat();
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

                params.print();
                fflush( stdout );
                status += ! params.okay();
                params.reset_output();
            }
            if (repeat > 1) {
                printf( "\n" );
            }
        } while(params.next());

        #ifdef BLAS_HAVE_PAPI
            if (params.papi() == 'y') {
                // stop papi
                require( PAPI_stop( event_set, counter_values ) == PAPI_OK );

                // print papi instrumentation
                blas::counter::print( (cset_list_object_t *)counter_values[0] );
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
