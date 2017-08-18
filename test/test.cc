#include <complex>

#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <omp.h>

#include "test.hh"

// -----------------------------------------------------------------------------
using libtest::ParamType;
using libtest::DataType;
using libtest::char2datatype;
using libtest::datatype2char;
using libtest::datatype2str;
using libtest::ansi_bold;
using libtest::ansi_red;
using libtest::ansi_normal;

// -----------------------------------------------------------------------------
enum Section {
    newline = 0,  // zero flag forces newline
    blas1,
    blas2,
    blas3,
};

const char* section_names[] = {
   "",  // none
   "Level 1 BLAS",
   "Level 2 BLAS",
   "Level 3 BLAS",
};

// { "", nullptr, Section::newline } entries force newline in help
std::vector< libtest::routines_t > routines = {
    // Level 1 BLAS
    { "asum",   test_asum,   Section::blas1   },
    { "axpy",   test_axpy,   Section::blas1   },
    { "copy",   test_copy,   Section::blas1   },
    { "dot",    test_dot,    Section::blas1   },
    { "dotu",   test_dotu,   Section::blas1   },
    { "iamax",  test_iamax,  Section::blas1   },
    { "nrm2",   test_nrm2,   Section::blas1   },
    { "scal",   test_scal,   Section::blas1   },
    { "swap",   test_swap,   Section::blas1   },

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

    // Level 3 BLAS
    { "gemm",   test_gemm,   Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "hemm",   test_hemm,   Section::blas3   },
    { "herk",   nullptr,     Section::blas3   },
    { "her2k",  nullptr,     Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "symm",   test_symm,   Section::blas3   },
    { "syrk",   nullptr,     Section::blas3   },
    { "syr2k",  nullptr,     Section::blas3   },
    { "",       nullptr,     Section::newline },

    { "trmm",   nullptr,     Section::blas3   },
    { "trsm",   nullptr,     Section::blas3   },
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
    //         name,       w,    type,             def, valid, help
    check     ( "check",   0,    ParamType::Value, 'y', "ny",  "check the results" ),
    ref       ( "ref",     0,    ParamType::Value, 'n', "ny",  "run reference; sometimes check -> ref" ),

    //          name,      w, p, type,             def, min,  max, help
    tol       ( "tol",     0, 0, ParamType::Value,  50,   1, 1000, "tolerance (e.g., error < tol*epsilon to pass)" ),
    repeat    ( "repeat",  0,    ParamType::Value,   1,   1, 1000, "times to repeat each test" ),
    verbose   ( "verbose", 0,    ParamType::Value,   0,   0,   10, "verbose level" ),
    cache     ( "cache",   0,    ParamType::Value,  20,   1, 1024, "total cache size, in MiB" ),

    // ----- routine parameters
    //          name,      w,    type,            def,                    char2enum,         enum2char,         enum2str,         help
    datatype  ( "type",    4,    ParamType::List, DataType::Double,       char2datatype,     datatype2char,     datatype2str,     "s=single (float), d=double, c=complex-single, z=complex-double" ),
    layout    ( "layout",  6,    ParamType::List, blas::Layout::ColMajor, blas::char2layout, blas::layout2char, blas::layout2str, "layout: r=row major, c=column major" ),
    side      ( "side",    6,    ParamType::List, blas::Side::Left,       blas::char2side,   blas::side2char,   blas::side2str,   "side: l=left, r=right" ),
    uplo      ( "uplo",    6,    ParamType::List, blas::Uplo::Lower,      blas::char2uplo,   blas::uplo2char,   blas::uplo2str,   "triangle: l=lower, u=upper" ),
    trans     ( "trans",   7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose: n=no-trans, t=trans, c=conj-trans" ),
    transA    ( "transA",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of A: n=no-trans, t=trans, c=conj-trans" ),
    transB    ( "transB",  7,    ParamType::List, blas::Op::NoTrans,      blas::char2op,     blas::op2char,     blas::op2str,     "transpose of B: n=no-trans, t=trans, c=conj-trans" ),
    diag      ( "diag",    7,    ParamType::List, blas::Diag::NonUnit,    blas::char2diag,   blas::diag2char,   blas::diag2str,   "diagonal: n=non-unit, u=unit" ),

    //          name,      w, p, type,            def,   min,     max, help
    dim       ( "dim",     6,    ParamType::List,          0, 1000000, "m x n x k dimensions" ),
    alpha     ( "alpha",   9, 4, ParamType::List,  pi,  -inf,     inf, "scalar alpha" ),
    beta      ( "beta",    9, 4, ParamType::List,   e,  -inf,     inf, "scalar beta" ),
    incx      ( "incx",    6,    ParamType::List,   1, -1000,    1000, "stride of x vector" ),
    incy      ( "incy",    6,    ParamType::List,   1, -1000,    1000, "stride of y vector" ),
    align     ( "align",   6,    ParamType::List,   1,     1,    1024, "column alignment (sets lda, ldb, etc. to multiple of align)" ),


    // ----- output parameters
    // min, max are ignored
    //          name,                  w, p, type,              def, min, max, help
    error     ( "SLATE\nerror",       11, 4, ParamType::Output, nan,   0,   0, "numerical error" ),
    ortho     ( "SLATE\north. error", 11, 4, ParamType::Output, nan,   0,   0, "orthogonality error" ),
    time      ( "SLATE\ntime (s)",    11, 4, ParamType::Output, nan,   0,   0, "time to solution" ),
    gflops    ( "SLATE\nGflop/s",     11, 4, ParamType::Output, nan,   0,   0, "Gflop/s rate" ),
    iters     ( "SLATE\niters",        6,    ParamType::Output,   0,   0,   0, "iterations to solution" ),

    ref_error ( "Ref.\nerror",        11, 4, ParamType::Output, nan,   0,   0, "reference numerical error" ),
    ref_ortho ( "Ref.\north. error",  11, 4, ParamType::Output, nan,   0,   0, "reference orthogonality error" ),
    ref_time  ( "Ref.\ntime (s)",     11, 4, ParamType::Output, nan,   0,   0, "reference time to solution" ),
    ref_gflops( "Ref.\nGflop/s",      11, 4, ParamType::Output, nan,   0,   0, "reference Gflop/s rate" ),
    ref_iters ( "Ref.\niters",         6,    ParamType::Output,   0,   0,   0, "reference iterations to solution" ),

    // default -1 means "no check"
    okay      ( "status",              6,    ParamType::Output,  -1,   0,   0, "success indicator" )
{
    // mark standard set of output fields as used
    okay  .value();
    error .value();
    time  .value();
    gflops.value();

    // mark framework parameters as used, so they will be accepted on the command line
    check  .value();
    tol    .value();
    repeat .value();
    verbose.value();
    cache  .value();

    // routine's parameters are marked by the test routine; see main
}

// -----------------------------------------------------------------------------
int main( int argc, char** argv )
{
    // Usage: test routine [params]
    // find routine to test
    if (argc < 2 ||
        strcmp( argv[1], "-h" ) == 0 ||
        strcmp( argv[1], "--help" ) == 0)
    {
        usage( argc, argv, routines, section_names );
        return 0;
    }
    const char* routine = argv[1];
    libtest::test_func_ptr test_routine = find_tester( routine, routines );
    if (test_routine == nullptr) {
        fprintf( stderr, "%s%sError: routine %s not found%s\n\n",
                 libtest::ansi_bold, libtest::ansi_red, routine,
                 libtest::ansi_normal );
        usage( argc, argv, routines, section_names );
        return -1;
    }

    // mark fields that are used (run=false)
    Params params;
    test_routine( params, false );

    // parse parameters after routine name
    params.parse( routine, argc-2, argv+2 );

    // print input so running `test [input] > out.txt` documents input
    printf( "input: %s", argv[0] );
    for (int i = 1; i < argc; ++i) {
        printf( " %s", argv[i] );
    }
    printf( "\n" );

    // run tests
    int status = 0;
    try {
        int repeat = params.repeat.value();
        libtest::DataType last = params.datatype.value();
        params.header();
        do {
            if (params.datatype.value() != last) {
                last = params.datatype.value();
                printf( "\n" );
            }
            for (int iter = 0; iter < repeat; ++iter) {
                test_routine( params, true );
                params.print();
                status += ! params.okay.value();
            }
            if (repeat > 1) {
                printf( "\n" );
            }
        } while( params.next() );
    }
    catch( blas::Error& e ) {
        status = 1;
        params.okay.value() = false;
        printf( "BLAS error: %s\n", e.what() );
        params.print();
    }

    if (status) {
        printf( "Some tests FAILED.\n" );
    }
    else {
        printf( "All tests passed.\n" );
    }
    return status;
}
