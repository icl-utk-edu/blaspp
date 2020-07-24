# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

from __future__ import print_function

import os
import re
import config
from   config import print_header, print_subhead, print_msg, print_warn, print_test, print_result
from   config import Error, get

#-------------------------------------------------------------------------------
def get_fortran_manglings():
    '''
    Returns list of flags to test different Fortran name manglings.
    Setting one or more of:
        fortran_mangling=add_, fortran_mangling=lower, fortran_mangling=upper
    limits which manglings are returned.

    Ex: get_fortran_manglings()
    returns ['-DFORTRAN_ADD_', '-DFORTRAN_LOWER', '-DFORTRAN_UPPER']
    '''
    # Warn about obsolete settings.
    if (config.environ['fortran_add_']):
        print_warn('Variable `fortran_add_`  is obsolete; use fortran_mangling=add_')
    if (config.environ['fortran_lower']):
        print_warn('Variable `fortran_lower` is obsolete; use fortran_mangling=lower')
    if (config.environ['fortran_upper']):
        print_warn('Variable `fortran_upper` is obsolete; use fortran_mangling=upper')

    # FORTRAN_ADD_, FORTRAN_LOWER, DFORTRAN_UPPER are BLAS++/LAPACK++.
    manglings = []
    fortran_mangling = config.environ['fortran_mangling'].lower()
    if ('add_'  in fortran_mangling):
        manglings.append('-DFORTRAN_ADD_')
    if ('lower' in fortran_mangling):
        manglings.append('-DFORTRAN_LOWER')
    if ('upper' in fortran_mangling):
        manglings.append('-DFORTRAN_UPPER')
    if (not manglings):
        cxx_actual = config.environ['CXX_actual']
        if (cxx_actual == 'xlc++'):
            # For IBM XL, change default mangling search order to lower, add_, upper,
            # ESSL includes all 3, but Netlib LAPACK has only one mangling.
            manglings = ['-DFORTRAN_LOWER',
                         '-DFORTRAN_ADD_',
                         '-DFORTRAN_UPPER']
        else:
            # For all others, mangling search order as add_, lower, upper,
            # since add_ is the most common.
            manglings = ['-DFORTRAN_ADD_',
                         '-DFORTRAN_LOWER',
                         '-DFORTRAN_UPPER']
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
    returns ['', '-DBLAS_ILP64 -DLAPACK_ILP64']
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
        int_sizes.append('-DBLAS_ILP64 -DLAPACK_ILP64')
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
                                ['-DFORTRAN_ADD_', '-DFORTRAN_LOWER'],
                                ['', '-DBLAS_ILP64'] )
    tests:
        CXX -Wall -DFORTRAN_ADD_               test.cc
        CXX -Wall -DFORTRAN_ADD_ -DBLAS_ILP64  test.cc
        CXX -Wall -DFORTRAN_LOWER              test.cc
        CXX -Wall -DFORTRAN_LOWER -DBLAS_ILP64 test.cc
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
# todo mkl_threaded, mkl_intel, mkl_gnu command line options.
def blas():
    '''
    Searches for BLAS in default libraries, MKL, ACML, ESSL, OpenBLAS,
    and Accelerate.
    Checks FORTRAN_ADD_, FORTRAN_LOWER, FORTRAN_UPPER.
    Checks int (LP64) and int64 (ILP64).
    Setting one or more of:
        blas=mkl, blas=acml, blas=essl, blas=openblas, blas=accelerate;
        blas_int=int, blas_int=int64;
        blas_threading=threaded, blas_threading=sequential;
        blas_fortran=gfortran, blas_fortran=ifort;
        fortran_mangling=add_, fortran_mangling=lower, fortran_mangling=upper;
    in the environment or on the command line, limits the search space.
    '''
    print_header( 'BLAS library' )
    print_msg( 'Also detects Fortran name mangling and BLAS integer size.' )

    # Warn about obsolete settings.
    if (config.environ['mkl']):
        print_warn('Variable `mkl`  is obsolete; use blas=mkl')
    if (config.environ['acml']):
        print_warn('Variable `acml` is obsolete; use blas=acml')
    if (config.environ['essl']):
        print_warn('Variable `essl` is obsolete; use blas=essl')
    if (config.environ['openblas']):
        print_warn('Variable `openblas` is obsolete; use blas=openblas')
    if (config.environ['accelerate']):
        print_warn('Variable `accelerate` is obsolete; use blas=accelerate')
    if (config.environ['lp64']):
        print_warn('Variable `lp64` is obsolete; use blas_int=int')
    if (config.environ['ilp64']):
        print_warn('Variable `ilp64` is obsolete; use blas_int=int64')

    BLAS_LIBRARIES = config.environ['BLAS_LIBRARIES']
    blas           = config.environ['blas'].lower()
    blas_fortran   = config.environ['blas_fortran'].lower()
    blas_int       = config.environ['blas_int'].lower()
    blas_threaded  = config.environ['blas_threaded'].lower()

    #---------------------------------------- BLAS_LIBRARIES
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

    #---------------------------------------- blas
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

    #---------------------------------------- blas_fortran
    test_gfortran = re.search( r'\b(gfortran)\b', blas_fortran ) is not None
    test_ifort    = re.search( r'\b(ifort)\b',    blas_fortran ) is not None
    if (not blas_fortran or blas_fortran == 'auto'):
        test_gfortran = True
        test_ifort    = True

    if (config.debug()):
        print( "blas_fortran        = '" + blas_fortran + "'\n"
             + "test_gfortran       = ", + test_gfortran,  "\n"
             + "test_ifort          = ", + test_ifort,     "\n" )

    #---------------------------------------- blas_int
    test_int   = re.search( r'\b(lp64|int|int32|int32_t)\b', blas_int ) is not None
    test_int64 = re.search( r'\b(ilp64|int64|int64_t)\b',    blas_int ) is not None
    if (not blas_int or blas_int == 'auto'):
        test_int   = True
        test_int64 = True

    if (config.debug()):
        print( "blas_int            = '" + blas_int + "'\n"
             + "test_int            = ", test_int,     "\n"
             + "test_int64          = ", test_int64,   "\n" )

    #---------------------------------------- blas_threaded
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

    #---------------------------------------- BLAS_LIBRARIES
    if (test_blas_libraries):
        choices.append(
            ['BLAS_LIBRARIES',
             {'LIBS': BLAS_LIBRARIES}] )
    # end

    #---------------------------------------- default; Cray libsci
    if (test_all or test_default):
        # Sometimes BLAS is in default libraries (e.g., on Cray).
        choices.append(
            ['Default',
             {}] )
    # end

    cxx        = config.environ['CXX']
    cxx_actual = config.environ['CXX_actual']
    has_openmp = config.environ['HAS_OPENMP']

    #---------------------------------------- Intel MKL
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
                         {'LIBS': '-lmkl_gf_ilp64 -lmkl_gnu_thread' + t_core,
                          'CXXFLAGS': '-DMKL_ILP64'}])

            elif (test_ifort and cxx_actual == 'icpc'):
                # Intel compiler + OpenMP: require intel_thread library.
                if (test_int):
                    choices_ifort.append(
                        ['Intel MKL (int, Intel Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_intel_lp64 -lmkl_intel_thread' + t_core}])
                if (test_int64):
                    choices_ifort.append(
                        ['Intel MKL (int64, Intel Fortran conventions, threaded)',
                         {'LIBS': '-lmkl_intel_ilp64 -lmkl_intel_thread' + t_core,
                          'CXXFLAGS': '-DMKL_ILP64'}])
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
                         {'LIBS': '-lmkl_intel_ilp64' + s_core,
                          'CXXFLAGS': '-DMKL_ILP64'}])
            # end

            if (test_gfortran):
                if (test_int):
                    choices_gfortran.append(
                        ['Intel MKL (int, GNU Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_gf_lp64' + s_core}])
                if (test_int64):
                    choices_gfortran.append(
                        ['Intel MKL (int64, GNU Fortran conventions, sequential)',
                         {'LIBS': '-lmkl_gf_ilp64' + s_core,
                          'CXXFLAGS': '-DMKL_ILP64'}])
            # end
        # end

        # For Intel icpc, prefer Intel fortran interfaces first;
        # otherwise,      prefer GNU   fortran interfaces first.
        if (cxx_actual == 'icpc'):
            choices.extend( choices_ifort )
            choices.extend( choices_gfortran )
        else:
            choices.extend( choices_gfortran )
            choices.extend( choices_ifort )
    # end mkl

    #---------------------------------------- IBM ESSL
    if (test_all or test_essl):
        if (test_threaded):
            if (test_int):
                choices.append(
                    ['IBM ESSL int (lp64), threaded',
                     {'LIBS': '-lesslsmp'}])
            if (test_int64):
                choices.append(
                    ['IBM ESSL int64 (ilp64), threaded',
                     {'LIBS': '-lesslsmp6464',
                      'CXXFLAGS': '-D_ESV6464'}])

        if (test_sequential):
            if (test_int):
                choices.append(
                    ['IBM ESSL int (lp64), sequential',
                     {'LIBS': '-lessl'}])
            if (test_int64):
                choices.append(
                    ['IBM ESSL int64 (ilp64), sequential',
                     {'LIBS': '-lessl6464',
                      'CXXFLAGS': '-D_ESV6464'}])
    # end essl

    #---------------------------------------- OpenBLAS
    if (test_all or test_openblas):
        choices.append(
            ['OpenBLAS',
             {'LIBS': '-lopenblas'}])
    # end

    #---------------------------------------- Apple Accelerate
    if (test_all or test_accelerate):
        # macOS puts cblas.h in weird places.
        paths = [
            '/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers',
            '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Headers',
        ]
        inc = ''
        for p in paths:
            if (os.path.exists( p + '/cblas.h' )):
                inc = '-I' + p + ' -DHAVE_ACCELERATE_CBLAS_H'
                break

        choices.append(
            ['MacOS Accelerate',
             {'LIBS': '-framework Accelerate',
              'CXXFLAGS': inc + '-DHAVE_ACCELERATE'}])
    # end

    #---------------------------------------- generic -lblas
    if (test_all or test_generic):
        choices.append(
            ['Generic BLAS',
             {'LIBS': '-lblas'}])
    # end

    #---------------------------------------- AMD ACML
    # Deprecated libraries last.
    if (test_all or test_acml):
        if (test_threaded):
            choices.append(
                ['AMD ACML (threaded)',
                 {'LIBS': '-lacml_mp'}])
        if (test_sequential):
            choices.append(
                ['AMD ACML (sequential)',
                 {'LIBS': '-lacml'}])
    # end

    #----------------------------------------
    # Test choices.
    manglings = get_fortran_manglings()
    int_sizes = get_int_sizes()
    passed = []
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
    config.environ.append( 'CXXFLAGS', '-DHAVE_BLAS' )
# end blas

#-------------------------------------------------------------------------------
def cblas():
    '''
    Searches for CBLAS library, first in already found BLAS library,
    then in -lcblas. Use blas() first to find BLAS library.
    '''
    print_header( 'CBLAS library' )
    choices = [
        ['CBLAS routines (cblas_ddot) available', {}],
        ['CBLAS routines (cblas_ddot) in -lcblas', {'LIBS': '-lcblas'}],
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
    config.environ.append( 'CXXFLAGS', '-DHAVE_CBLAS' )
# end cblas

#-------------------------------------------------------------------------------
def test_lapack( src, label, append=False ):
    '''
    Used by lapack() and lapack_uncommon() to search for LAPACK routines,
    first in already found libraries, then in -llapack.
    Rechecks Fortran name mangling if -llapack exists but linking fails.
    '''
    # try in BLAS library
    print_test( label + ' available' )
    (rc, out, err) = config.compile_obj( src )
    if (rc != 0):
        raise Error

    (rc, out, err) = config.link_exe( src )
    if (rc == 0):
        (rc, out, err) = config.run_exe( src )
        print_result( 'label', rc )
        if (rc != 0):
            raise Error( 'LAPACK linked, but failed to run' )
        # Otherwise, success! See also lapack().
    else:
        # Default failed, try with -llapack
        print_result( 'label', rc )
        print_test( label + ' in -llapack' )
        env = {'LIBS': '-llapack'}
        (rc, out, err) = config.compile_exe( 'config/hello.cc', env )
        if (rc == 0):
            # -llapack exists
            (rc, out, err) = config.compile_obj( src, env )
            if (rc != 0):
                raise Error( 'Unexpected error: ' + src + ' failed to compile' )

            (rc, out, err) = config.link_exe( src, env )
            if (rc == 0):
                # -llapack linked
                (rc, out, err) = config.run_exe( src, env )
                print_result( 'label', rc )
                if (rc == 0):
                    # Success! -llapack worked. See also lapack().
                    if (append):
                        config.environ.append( 'LIBS', env['LIBS'] )
                    else:
                        config.environ.merge( env )
                else:
                    raise Error( 'LAPACK linked -llapack, but failed to run' )
                # end
            else:
                print_result( 'label', rc )
                # -llapack exists but didn't link
                print_subhead( '-llapack exists, but linking failed. Re-checking Fortran mangling.' )
                # Get, then undef, original mangling & int_sizes.
                cxxflags = config.environ['CXXFLAGS']
                old_mangling_sizes = re.findall(
                    r'-D(FORTRAN_(?:ADD_|LOWER|UPPER)|\w*ILP64)\b', cxxflags )
                config.environ['CXXFLAGS'] = re.sub(
                    r'-D(FORTRAN_(?:ADD_|LOWER|UPPER)|\w*ILP64|ADD_|NOCHANGE|UPCASE)\b',
                    r'', cxxflags )
                manglings = get_fortran_manglings()
                int_sizes = get_int_sizes()
                (rc, out, err, env) = compile_with_manglings(
                    src, env, manglings, int_sizes )
                if (rc == 0):
                    (rc, out, err) = config.compile_run( 'config/blas.cc', env,
                        'Re-checking Fortran mangling for BLAS' )
                    if (rc == 0):
                        # Success! See also lapack().
                        new_mangling_sizes = re.findall(
                            r'-D(FORTRAN_(?:ADD_|LOWER|UPPER)|\w*ILP64)\b',
                            env['CXXFLAGS'])
                        print( font.red(
                               'Changing Fortran name mangling for both BLAS and LAPACK to '
                               + ' '.join( new_mangling_sizes ) ) )
                        config.environ.merge( env )

                    else:
                        raise Error( 'BLAS and LAPACK require different Fortran name manglings' )
                    # end
                else:
                    raise Error( 'No Fortran name mangling worked for LAPACK (seems odd).' )
                # end
            # end
        else:
            # -llapack doesn't exist
            print_result( 'label', rc )
            raise Error( 'LAPACK not found' )
        # end
    # end
# end test_lapack

#-------------------------------------------------------------------------------
def lapack():
    '''
    Search for common LAPACK routines, first in already found libraries,
    then in -llapack.
    '''
    print_header( 'LAPACK library' )
    test_lapack( 'config/lapack_potrf.cc', 'LAPACK routines (dpotrf)' )
    # no error thrown: success!
    config.environ.append( 'CXXFLAGS', '-DHAVE_LAPACK' )
# end lapack

#-------------------------------------------------------------------------------
def lapack_uncommon():
    '''
    Search for uncommon LAPACK routines, first in already found libraries,
    then in -llapack.

    This is needed because some libraries, like ESSL and ATLAS,
    include part but not all of LAPACK, so need -llapack added.
    Cholesky with pivoting (pstrf) is a routine from LAPACK >= 3.2 that is
    unlikely to be included in a subset of LAPACK routines.
    '''
    test_lapack( 'config/lapack_pstrf.cc', 'Uncommon routines (dpstrf)', append=True )
# end lapack_uncommon

#-------------------------------------------------------------------------------
def lapacke():
    '''
    Search for LAPACKE in existing BLAS/LAPACK libraries,
    found with blas() and lapack(), then in -llapacke.
    '''
    print_header( 'LAPACKE library' )
    choices = [
        ['LAPACKE routines (LAPACKE_dpotrf) available', {}],
        ['LAPACKE routines (LAPACKE_dpotrf) in -llapacke',
            {'LIBS': '-llapacke'}],
    ]

    passed = []
    for (label, env) in choices:
        (rc, out, err) = config.compile_run( 'config/lapacke_potrf.cc', env, label )
        if (rc == 0):
            passed.append( (label, env) )
            break
    # end

    labels = map( lambda c: c[0], passed )
    i = config.choose( 'Choose LAPACKE library:', labels )
    config.environ.merge( passed[i][1] )
    config.environ.append( 'CXXFLAGS', '-DHAVE_LAPACKE' )
# end lapacke

#-------------------------------------------------------------------------------
def lapacke_uncommon():
    '''
    ESSL doesn't include all of LAPACKE, so needs -llapacke added.
    Cholesky with pivoting (LAPACKE_pstrf) is one from LAPACK >= 3.4 that ESSL
    excludes.
    '''
    choices = [
        ('Uncommon routines (LAPACKE_dpstrf) available', {}),
    ]
    if ('-llapacke' not in config.environ['LIBS']):
        choices.append( ['Uncommon routines (LAPACKE_dpstrf) in -llapacke',
                            {'LIBS': '-llapacke'}] )

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
# end lapacke_uncommon

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
        config.environ.append( 'CXXFLAGS', '-DHAVE_F2C' )
    else:
        print_warn( 'unexpected error!' )
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
        config.environ.append( 'CXXFLAGS', '-DBLAS_COMPLEX_RETURN_ARGUMENT' )
    else:
        print_warn( 'unexpected error!' )
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
        config.environ.append( 'CXXFLAGS', '-DLAPACK_VERSION=%s' % v )
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
                                         'LAPACK XBLAS routines (dposvxx) available' )
    if (rc == 0):
        config.environ.append( 'CXXFLAGS', '-DHAVE_XBLAS' )
# end

#-------------------------------------------------------------------------------
def lapack_matgen():
    '''
    Search for LAPACK matrix generation routines (tmglib)
    in found BLAS/LAPACK libraries, then in -llapacke.
    '''
    choices = [
        ['Matrix generation routines (lagsy) available', {}],
        ['Matrix generation routines (lagsy) in -ltmglib',
            {'LIBS': '-ltmglib'}],
    ]

    passed = []
    for (label, env) in choices:
        (rc, out, err) = config.compile_run( 'config/lapack_matgen.cc', env, label )
        if (rc == 0):
            config.environ.merge( env )
            config.environ.append( 'CXXFLAGS', '-DHAVE_MATGEN' )
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
        config.environ.append( 'CXXFLAGS', '-DHAVE_MKL' )
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
        config.environ.append( 'CXXFLAGS', '-DHAVE_ACML' )
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
        config.environ.append( 'CXXFLAGS', '-DHAVE_ESSL' )
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
        config.environ.append( 'CXXFLAGS', '-DHAVE_OPENBLAS' )
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
