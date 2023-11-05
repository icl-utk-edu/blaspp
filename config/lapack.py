# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

from __future__ import print_function

import os
import re
import config
from   config import print_header, print_subhead, print_msg, print_warn, \
                     print_test, print_result, define, Error, get

#-------------------------------------------------------------------------------
def get_fortran_manglings():
    '''
    Returns list of flags to test different Fortran name manglings.
    Setting one or more of:
        fortran_mangling=add_, fortran_mangling=lower, fortran_mangling=upper
    limits which manglings are returned.

    Ex: get_fortran_manglings()
    returns ['-D<name>_FORTRAN_ADD_', '-D<name>_FORTRAN_LOWER', '-D<name>_FORTRAN_UPPER']
    '''
    # FORTRAN_ADD_, FORTRAN_LOWER, DFORTRAN_UPPER are BLAS++/LAPACK++.
    manglings = []
    fortran_mangling = config.environ['fortran_mangling'].lower()
    if ('add_'  in fortran_mangling):
        manglings.append( define('FORTRAN_ADD_') )
    if ('lower' in fortran_mangling):
        manglings.append( define('FORTRAN_LOWER') )
    if ('upper' in fortran_mangling):
        manglings.append( define('FORTRAN_UPPER') )
    if (not manglings):
        cxx_actual = config.environ['CXX_actual']
        if (cxx_actual == 'xlc++'):
            # For IBM XL, change default mangling search order to lower, add_, upper,
            # ESSL includes all 3, but Netlib LAPACK has only one mangling.
            manglings = [define('FORTRAN_LOWER'),
                         define('FORTRAN_ADD_'),
                         define('FORTRAN_UPPER')]
        else:
            # For all others, mangling search order as add_, lower, upper,
            # since add_ is the most common.
            manglings = [define('FORTRAN_ADD_'),
                         define('FORTRAN_LOWER'),
                         define('FORTRAN_UPPER')]
    return manglings
# end

#-------------------------------------------------------------------------------
def get_int_sizes():
    '''
    Returns list of flags to test different integer sizes.
    Setting one or more of:
        blas_int=int
        blas_int=int64
    limits which sizes are returned.

    Ex: get_int_sizes()
    returns ['', '-D<name>_ILP64']
    where '' is compiler's default, usually 32-bit int in LP64.
    '''
    # todo: repeated from below
    blas_int = config.environ['blas_int'].lower()
    test_int   = re.search( r'\b(lp64|int|int32|int32_t)\b', blas_int ) is not None
    test_int64 = re.search( r'\b(ilp64|int64|int64_t)\b',    blas_int ) is not None
    if (not blas_int or blas_int == 'auto'):
        test_int   = True
        test_int64 = True

    int_sizes = []
    if (test_int):
        int_sizes.append('') # i.e., default int
    if (test_int64):
        int_sizes.append( define('ILP64') )
    return int_sizes
# end

#-------------------------------------------------------------------------------
def compile_with_manglings( src, env, manglings, int_sizes ):
    '''
    Tries to compile, link, and run source file src with each of the given
    manglings and integer sizes.
    Returns (returncode, stdout, stderr, env_copy)
    from either the successful run or the last unsuccessful run.

    Ex: compile_with_manglings( 'test.cc', {'CXXFLAGS': '-Wall'},
                                ['-D<name>_FORTRAN_ADD_', '-D<name>_FORTRAN_LOWER'],
                                ['', '-D<name>_ILP64'] )
    tests:
        CXX -Wall -D<name>_FORTRAN_ADD_                 test.cc
        CXX -Wall -D<name>_FORTRAN_ADD_ -D<name>_ILP64  test.cc
        CXX -Wall -D<name>_FORTRAN_LOWER                test.cc
        CXX -Wall -D<name>_FORTRAN_LOWER -D<name>_ILP64 test.cc
    '''
    rc = -1
    for mangling in manglings:
        for size in int_sizes:
            print_test( '    ' + mangling +' '+ size )
            # modify a copy to save in passed
            env2 = env.copy()
            env2['CXXFLAGS'] = get(env2, 'CXXFLAGS') +' '+ mangling +' '+ size
            (rc_link, out, err) = config.compile_exe( 'config/hello.cc', env2 )
            # if hello didn't link, assume library not found
            if (rc_link != 0):
                print_result( 'label', rc_link )
                break

            (rc, out, err) = config.compile_exe( src, env2 )
            # if int32 didn't link, int64 won't either
            if (rc != 0):
                print_result( 'label', rc )
                break

            # if int32 runs, skip int64
            (rc, out, err) = config.run_exe( src )
            print_result( 'label', rc )
            if (rc == 0):
                break
        # end
        # break if library not found or on first mangling that works
        if (rc_link != 0 or rc == 0):
            break
    # end
    return (rc, out, err, env2)
# end

#-------------------------------------------------------------------------------
def blas():
    '''
    Searches for BLAS in default libraries, MKL, ACML, ESSL, OpenBLAS,
    and Accelerate.
    Checks FORTRAN_ADD_, FORTRAN_LOWER, FORTRAN_UPPER.
    Checks int (LP64) and int64 (ILP64).
    Setting one or more of:
        blas = {mkl, acml, essl, openblas, accelerate, generic};
        blas_int = {int, int64};
        blas_threaded = {y, n};
        blas_fortran = {gfortran, ifort};
        fortran_mangling = {add_, lower, upper}
    in the environment or on the command line, limits the search space.
    '''
    print_header( 'BLAS library' )
    print_msg( 'Also detects Fortran name mangling and BLAS integer size.' )

    #----------------------------------------
    # Parse options.
    BLAS_LIBRARIES = config.environ['BLAS_LIBRARIES']
    blas           = config.environ['blas'].lower()
    blas_fortran   = config.environ['blas_fortran'].lower()
    blas_int       = config.environ['blas_int'].lower()
    blas_threaded  = config.environ['blas_threaded'].lower()

    #-------------------- BLAS_LIBRARIES
    # If testing BLAS_LIBRARIES, ignore other flags (blas, ...).
    test_blas_libraries = (BLAS_LIBRARIES != '')
    if (test_blas_libraries):
        blas          = 'none'
        blas_fortran  = ''
        blas_int      = ''
        blas_threaded = ''

    if (config.debug()):
        print( "BLAS_LIBRARIES      = '" + BLAS_LIBRARIES  + "'\n"
             + "test_blas_libraries = ", test_blas_libraries, "\n" )

    #-------------------- blas
    test_all        = (not blas or blas == 'auto')
    test_acml       = re.search( r'\b(acml)\b',                blas ) is not None
    test_accelerate = re.search( r'\b(apple|accelerate)\b',    blas ) is not None
    test_default    = re.search( r'\b(cray|libsci|default)\b', blas ) is not None
    test_essl       = re.search( r'\b(ibm|essl)\b',            blas ) is not None
    test_mkl        = re.search( r'\b(intel|mkl)\b',           blas ) is not None
    test_openblas   = re.search( r'\b(openblas)\b',            blas ) is not None
    test_generic    = re.search( r'\b(generic)\b',             blas ) is not None

    if (config.debug()):
        print( "blas                = '" + blas            + "'\n"
             + "test_acml           = ", test_acml,           "\n"
             + "test_accelerate     = ", test_accelerate,     "\n"
             + "test_default        = ", test_default,        "\n"
             + "test_essl           = ", test_essl,           "\n"
             + "test_mkl            = ", test_mkl,            "\n"
             + "test_openblas       = ", test_openblas,       "\n"
             + "test_generic        = ", test_generic,        "\n"
             + "test_all            = ", test_all,            "\n" )

    #-------------------- blas_fortran
    test_gfortran = re.search( r'\b(gfortran)\b', blas_fortran ) is not None
    test_ifort    = re.search( r'\b(ifort)\b',    blas_fortran ) is not None
    if (not blas_fortran or blas_fortran == 'auto'):
        test_gfortran = True
        test_ifort    = True

    if (config.debug()):
        print( "blas_fortran        = '" + blas_fortran + "'\n"
             + "test_gfortran       = ", + test_gfortran,  "\n"
             + "test_ifort          = ", + test_ifort,     "\n" )

    #-------------------- blas_int
    test_int   = re.search( r'\b(lp64|int|int32|int32_t)\b', blas_int ) is not None
    test_int64 = re.search( r'\b(ilp64|int64|int64_t)\b',    blas_int ) is not None
    if (not blas_int or blas_int == 'auto'):
        test_int   = True
        test_int64 = True

    if (config.debug()):
        print( "blas_int            = '" + blas_int + "'\n"
             + "test_int            = ", test_int,     "\n"
             + "test_int64          = ", test_int64,   "\n" )

    #-------------------- blas_threaded
    test_threaded   = re.search( r'\b(y|yes|true|on|1)\b',  blas_threaded ) is not None
    test_sequential = re.search( r'\b(n|no|false|off|0)\b', blas_threaded ) is not None
    if (not blas_threaded or blas_threaded == 'auto'):
        test_threaded   = True
        test_sequential = True

    if (config.debug()):
        print( "blas_threaded       = '" + blas_threaded + "'\n"
             + "test_threaded       = ", test_threaded,     "\n"
             + "test_sequential     = ", test_sequential,   "\n" )

    #----------------------------------------
    # Build list of libraries to check.
    choices = []

    cxx        = config.environ['CXX']
    cxx_actual = config.environ['CXX_actual']
    has_openmp = config.environ['HAS_OPENMP']

    #-------------------- BLAS_LIBRARIES
    if (test_blas_libraries):
        choices.append( ['BLAS_LIBRARIES', {'LIBS': BLAS_LIBRARIES}] )

    #-------------------- default; Cray libsci
    if (test_all or test_default):
        # Sometimes BLAS is in default libraries (e.g., on Cray).
        choices.append( ['Default', {}] )

    #-------------------- Intel MKL
    if (test_all or test_mkl):
        choices_ifort    = []
        choices_gfortran = []
        if (test_threaded and has_openmp):
            t_core = ' -lmkl_core -lm'
            if (test_gfortran and cxx_actual == 'g++'):
                # GNU compiler + OpenMP: require gnu_thread library.
                if (test_int):
                    choices_gfortran.append(
                        ['Intel MKL (int, GNU Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_gf_lp64 -lmkl_gnu_thread' + t_core}])
                if (test_int64):
                    choices_gfortran.append(
                        ['Intel MKL (int64, GNU Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_gf_ilp64 -lmkl_gnu_thread' + t_core}])

            elif (test_ifort and cxx_actual in ('icpc', 'icpx')):
                # Intel compiler + OpenMP: require intel_thread library.
                if (test_int):
                    choices_ifort.append(
                        ['Intel MKL (int, Intel Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_intel_lp64 -lmkl_intel_thread' + t_core}])
                if (test_int64):
                    choices_ifort.append(
                        ['Intel MKL (int64, Intel Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_intel_ilp64 -lmkl_intel_thread' + t_core}])
            else:
                # MKL doesn't have libraries for other OpenMP backends.
                print( "Skipping threaded MKL for non-GNU, non-Intel compiler" )
        # end threaded

        if (test_sequential):
            s_core = ' -lmkl_sequential -lmkl_core -lm'
            if (test_ifort):
                if (test_int):
                    choices_ifort.append(
                        ['Intel MKL (int, Intel Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_intel_lp64' + s_core}])
                if (test_int64):
                    choices_ifort.append(
                        ['Intel MKL (int64, Intel Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_intel_ilp64' + s_core}])
            # end

            if (test_gfortran):
                if (test_int):
                    choices_gfortran.append(
                        ['Intel MKL (int, GNU Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_gf_lp64' + s_core}])
                if (test_int64):
                    choices_gfortran.append(
                        ['Intel MKL (int64, GNU Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_gf_ilp64' + s_core}])
            # end
        # end

        # For Intel compilers, prefer Intel fortran interfaces first;
        # otherwise,           prefer GNU   fortran interfaces first.
        if (cxx_actual in ('icpc', 'icpx')):
            choices.extend( choices_ifort )
            choices.extend( choices_gfortran )
        else:
            choices.extend( choices_gfortran )
            choices.extend( choices_ifort )
    # end mkl

    #-------------------- IBM ESSL
    if (test_all or test_essl):
        if (test_threaded):
            if (test_int):
                choices.append(
                    ['IBM ESSL int (lp64), threaded',
                     {'LIBS': '-lesslsmp'}])
            if (test_int64):
                choices.append(
                    ['IBM ESSL int64 (ilp64), threaded',
                     {'LIBS': '-lesslsmp6464'}])

        if (test_sequential):
            if (test_int):
                choices.append(
                    ['IBM ESSL int (lp64), sequential',
                     {'LIBS': '-lessl'}])
            if (test_int64):
                choices.append(
                    ['IBM ESSL int64 (ilp64), sequential',
                     {'LIBS': '-lessl6464'}])
    # end essl

    #-------------------- OpenBLAS
    if (test_all or test_openblas):
        choices.append( ['OpenBLAS', {'LIBS': '-lopenblas'}])

    #-------------------- Apple Accelerate
    if (test_all or test_accelerate):
        # macOS puts cblas.h in weird places.
        paths = [
            '/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers',
            '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers',
        ]
        inc = ''
        for p in paths:
            if (os.path.exists( p + '/cblas.h' )):
                inc = '-I' + p + ' ' + define('HAVE_ACCELERATE_CBLAS_H')
                break

        choices.append(
            ['MacOS Accelerate',
             {'LIBS': '-framework Accelerate',
              'CXXFLAGS': inc + define('HAVE_ACCELERATE')}])
    # end

    #-------------------- generic -lblas
    if (test_all or test_generic):
        choices.append( ['Generic BLAS', {'LIBS': '-lblas'}])

    #-------------------- AMD ACML
    # Deprecated libraries last.
    if (test_all or test_acml):
        if (test_threaded):
            choices.append( ['AMD ACML (threaded)', {'LIBS': '-lacml_mp'}])
        if (test_sequential):
            choices.append( ['AMD ACML (sequential)', {'LIBS': '-lacml'}])
    # end

    #----------------------------------------
    # Test choices.
    manglings = get_fortran_manglings()
    int_sizes = get_int_sizes()
    passed = []
    print_subhead( 'BLAS (ddot) in:' )
    for (label, env) in choices:
        title = label
        if ('LIBS' in env):
            title += '\n    ' + env['LIBS']
        print_subhead( title )
        (rc, out, err, env2) = compile_with_manglings(
            'config/blas.cc', env, manglings, int_sizes )
        if (rc == 0):
            passed.append( (label, env2) )
            if (not config.interactive()):
                break
    # end

    labels = map( lambda c: c[0], passed )
    i = config.choose( 'Choose BLAS library:', labels )
    config.environ.merge( passed[i][1] )
# end blas

#-------------------------------------------------------------------------------
def cblas():
    '''
    Searches for CBLAS library, first in already found BLAS library,
    then in -lcblas. Use blas() first to find BLAS library.
    '''
    print_header( 'CBLAS library' )
    choices = [
        ['CBLAS (cblas_ddot) in BLAS library', {}],
        ['CBLAS (cblas_ddot) in -lcblas', {'LIBS': '-lcblas'}],
    ]

    passed = []
    for (label, env) in choices:
        (rc, out, err) = config.compile_run( 'config/cblas.cc', env, label )
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0], passed )
    i = config.choose( 'Choose CBLAS library:', labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', define('HAVE_CBLAS') )
# end cblas

#-------------------------------------------------------------------------------
# This code is structured similarly to blas().
def lapack():
    '''
    Search for LAPACK library, first in already found BLAS libraries,
    then in -llapack. Use blas() first to find BLAS library.
    This checks for `pstrf` to ensure we're getting a complete LAPACK,
    since `pstrf` has been in LAPACK for a long time, but is omitted
    from some libraries like ESSL and ATLAS that contain only selected
    routines like `potrf`.
    '''
    print_header( 'LAPACK library' )

    #----------------------------------------
    # Parse options.
    LAPACK_LIBRARIES = config.environ['LAPACK_LIBRARIES']
    lapack = config.environ['lapack'].lower()

    #-------------------- LAPACK_LIBRARIES
    # If testing LAPACK_LIBRARIES, ignore other flags (lapack, ...).
    test_lapack_libraries = (LAPACK_LIBRARIES != '')
    if (test_lapack_libraries):
        lapack = 'none'

    if (config.debug()):
        print( "LAPACK_LIBRARIES      = '" + LAPACK_LIBRARIES  + "'\n"
             + "test_lapack_libraries = ", test_lapack_libraries, "\n" )

    #-------------------- lapack
    test_all     = (not lapack or lapack == 'auto')
    test_default = re.search( r'\b(default)\b', lapack ) is not None
    test_generic = re.search( r'\b(generic)\b', lapack ) is not None

    if (config.debug()):
        print( "lapack              = '" + lapack          + "'\n"
             + "test_default        = ", test_default,        "\n"
             + "test_generic        = ", test_generic,        "\n"
             + "test_all            = ", test_all,            "\n" )

    #----------------------------------------
    # Build list of libraries to check.
    choices = []

    #-------------------- LAPACK_LIBRARIES
    if (test_lapack_libraries):
        choices.append( ['LAPACK_LIBRARIES = ' + LAPACK_LIBRARIES,
                         {'LIBS': LAPACK_LIBRARIES}] )

    #-------------------- default (e.g., in BLAS library)
    if (test_all or test_default):
        choices.append( ['BLAS library', {}] )

    #-------------------- generic -llapack
    if (test_all or test_generic):
        choices.append( ['generic -llapack', {'LIBS': '-llapack'}])

    #----------------------------------------
    # Test choices.
    passed = []
    for (label, env) in choices:
        label = 'LAPACK (dpstrf) in ' + label
        (rc, out, err) = config.compile_run(
            'config/lapack_pstrf.cc', env, label )
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0], passed )
    i = config.choose( 'Choose LAPACK library:', labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', define('HAVE_LAPACK') )
# end lapack

#-------------------------------------------------------------------------------
def lapacke():
    '''
    Search for LAPACKE in existing BLAS/LAPACK libraries,
    found with blas() and lapack(), then in -llapacke.
    '''
    print_header( 'LAPACKE library' )
    choices = [
        ['LAPACKE (LAPACKE_dpotrf) in LAPACK library', {}],
        ['LAPACKE (LAPACKE_dpotrf) in -llapacke',
            {'LIBS': '-llapacke'}],
    ]

    passed = []
    for (label, env) in choices:
        (rc, out, err) = config.compile_run( 'config/lapacke_pstrf.cc', env, label )
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0], passed )
    i = config.choose( 'Choose LAPACKE library:', labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', define('HAVE_LAPACKE') )
# end lapacke

#-------------------------------------------------------------------------------
def blas_float_return():
    '''
    Normally, float functions like sdot return float.
    f2c and g77 always returned double, even for float functions like sdot.
    This affects clapack and MacOS Accelerate.
    '''
    (rc, out, err) = config.compile_run(
        'config/return_float.cc', {},
        'BLAS (sdot) returns float as float (standard)' )
    if (rc == 0):
        return

    (rc, out, err) = config.compile_run(
        'config/return_float_f2c.cc', {},
        'BLAS (sdot) returns float as double (f2c convention)' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', define('HAVE_F2C') )
    else:
        raise Error( "Could not determine return type of sdot; check log." )
# end

#-------------------------------------------------------------------------------
def blas_complex_return():
    '''
    For complex valued functions like zdotc, GNU returns complex, while
    Intel ifort and f2c return the complex in a hidden first argument.
    '''
    (rc, out, err) = config.compile_run(
        'config/return_complex.cc', {},
        'BLAS (zdotc) returns complex (GNU gfortran convention)' )
    if (rc == 0):
        return

    (rc, out, err) = config.compile_run(
        'config/return_complex_argument.cc', {},
        'BLAS (zdotc) returns complex as hidden argument (Intel ifort convention)' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', define('COMPLEX_RETURN_ARGUMENT') )
    else:
        raise Error( "Could not determine how zdot returns complex result; check log." )
# end

#-------------------------------------------------------------------------------
def lapack_version():
    '''
    Check for LAPACK version using ilaver().
    '''
    config.print_test( 'LAPACK version' )
    (rc, out, err) = config.compile_run( 'config/lapack_version.cc' )
    s = re.search( r'^LAPACK_VERSION=((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        v = '%d%02d%02d' % (int(s.group(2)), int(s.group(3)), int(s.group(4)))
        config.environ.append( 'CXXFLAGS', define('LAPACK_VERSION', v) )
        config.print_result( 'LAPACK', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'LAPACK', rc )
# end

#-------------------------------------------------------------------------------
def lapack_xblas():
    '''
    Check for LAPACK routines that use XBLAS in found BLAS/LAPACK libraries.
    '''
    (rc, out, err) = config.compile_run( 'config/lapack_xblas.cc', {},
                                         'LAPACK XBLAS (dposvxx) in LAPACK library' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', define('HAVE_XBLAS') )
# end

#-------------------------------------------------------------------------------
def lapack_matgen():
    '''
    Search for LAPACK matrix generation routines (tmglib)
    in found BLAS/LAPACK libraries, then in -llapacke.
    '''
    choices = [
        ['Matrix generation (dlagsy) in LAPACK library', {}],
        ['Matrix generation (dlagsy) in -ltmglib',
            {'LIBS': '-ltmglib'}],
    ]

    passed = []
    for (label, env) in choices:
        (rc, out, err) = config.compile_run( 'config/lapack_matgen.cc', env, label )
        if (rc == 0):
            config.environ.merge( env )
            config.environ.append( 'CXXFLAGS', define('HAVE_MATGEN') )
            break
    # end
# end

#-------------------------------------------------------------------------------
def mkl_version():
    '''
    Check for MKL version via MKL_Get_Version().
    '''
    config.print_test( 'MKL version' )
    (rc, out, err) = config.compile_run( 'config/mkl_version.cc' )
    s = re.search( r'^MKL_VERSION=((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', define('HAVE_MKL') )
        config.print_result( 'MKL', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'MKL', rc )
# end

#-------------------------------------------------------------------------------
def acml_version():
    '''
    Check for ACML version via acmlversion().
    '''
    config.print_test( 'ACML version' )
    (rc, out, err) = config.compile_run( 'config/acml_version.cc' )
    s = re.search( r'^ACML_VERSION=((\d+)\.(\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', define('HAVE_ACML') )
        config.print_result( 'ACML', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'ACML', rc )
# end

#-------------------------------------------------------------------------------
def essl_version():
    '''
    Check for ESSL version via iessl().
    '''
    config.print_test( 'ESSL version' )
    (rc, out, err) = config.compile_run( 'config/essl_version.cc' )
    s = re.search( r'^ESSL_VERSION=((\d+)\.(\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', define('HAVE_ESSL') )
        config.print_result( 'ESSL', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'ESSL', rc )
# end

#-------------------------------------------------------------------------------
def openblas_version():
    '''
    Check for OpenBLAS version via OPENBLAS_VERSION constant.
    '''
    config.print_test( 'OpenBLAS version' )
    (rc, out, err) = config.compile_run( 'config/openblas_version.cc' )
    s = re.search( r'^OPENBLAS_VERSION=.*?((\d+)\.(\d+)\.(\d+))', out )
    if (rc == 0 and s):
        config.environ.append( 'CXXFLAGS', define('HAVE_OPENBLAS') )
        config.print_result( 'OpenBLAS', rc, '(' + s.group(1) + ')' )
    else:
        config.print_result( 'OpenBLAS', rc )
# end

#-------------------------------------------------------------------------------
def vendor_version():
    '''
    Check for MKL, ACML, ESSL, or OpenBLAS version number in BLAS/LAPACK
    libraries.
    '''
    # If we can, be smart looking for MKL, ESSL, or OpenBLAS version;
    # otherwise, check them all.
    LIBS = config.environ['LIBS']
    if ('-lmkl' in LIBS):
        mkl_version()
    elif ('-lacml' in LIBS):
        acml_version()
    elif ('-lessl' in LIBS):
        essl_version()
    elif ('-lopenblas' in LIBS):
        openblas_version()
    elif ('-framework Accelerate' in LIBS):
        pass
    else:
        mkl_version()
        acml_version()
        essl_version()
        openblas_version()
    # end
# end
