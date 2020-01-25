from __future__ import print_function

import os
import re
import config
from   config import ansi_bold, ansi_red, ansi_normal
from   config import print_header, print_subhead, print_msg, print_warn, print_test, print_result
from   config import get
from   config import Error

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

    # ADD_, NOCHANGE, UPCASE are traditional in lapack
    # FORTRAN_ADD_, FORTRAN_LOWER, DFORTRAN_UPPER are BLAS++/LAPACK++.
    manglings = []
    if ('add_'  in config.environ['fortran_mangling']):
        manglings.append('-DFORTRAN_ADD_ -DADD_')
    if ('lower' in config.environ['fortran_mangling']):
        manglings.append('-DFORTRAN_LOWER -DNOCHANGE')
    if ('upper' in config.environ['fortran_mangling']):
        manglings.append('-DFORTRAN_UPPER -DUPCASE')
    if (not manglings):
        if (config.environ['fortran_mangling']):
            print_warn('Unknown fortran_mangling: '+ config.environ['fortran_mangling'])
        manglings = ['-DFORTRAN_ADD_ -DADD_',
                     '-DFORTRAN_LOWER -DNOCHANGE',
                     '-DFORTRAN_UPPER -DUPCASE']
    return manglings
# end

#-------------------------------------------------------------------------------
def get_int_sizes():
    '''
    Returns list of flags to test different integer sizes.
    Setting one or more of:
        lp64=1, ilp64=1
    limits which sizes are returned.

    Ex: get_int_sizes()
    returns ['', '-DBLAS_ILP64']
    where '' is compiler's default, usually 32-bit int in LP64.
    '''
    int_sizes = []
    if (config.environ['lp64'] == '1'):
        int_sizes.append('') # i.e., default int
    if (config.environ['ilp64'] == '1'):
        int_sizes.append('-DBLAS_ILP64')
    if (not int_sizes):
        int_sizes = ['', '-DBLAS_ILP64']
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
    Checks int (LP64) and int64_t (ILP64).
    Setting one or more of:
        blas=mkl, blas=acml, blas=essl, blas=openblas, blas=accelerate;
        fortran_mangling=add_, fortran_mangling=lower, fortran_mangling=upper;
        lp64=1, ilp64=1
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

    test_mkl        = ('mkl'        in config.environ['blas'])
    test_acml       = ('acml'       in config.environ['blas'])
    test_essl       = ('essl'       in config.environ['blas'])
    test_openblas   = ('openblas'   in config.environ['blas'])
    test_accelerate = ('accelerate' in config.environ['blas'])
    # otherwise, test all
    test_all = not (test_mkl or test_acml or test_essl or test_openblas or
                    test_accelerate)

    # build list of choices to test
    choices = []

    if (test_all):
        # sometimes BLAS is in default libraries (e.g., on Cray)
        choices.extend([
            ['Default', {}],
        ])
    # end

    if (test_all or test_mkl):
        choices.extend([
            # each pair has Intel conventions, then GNU conventions
            # int, threaded
            ['Intel MKL (int, Intel conventions, threaded)',
                {'LIBS':     '-lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp',
                 'LDFLAGS':  '-fopenmp'}],
            ['Intel MKL (int, GNU conventions, threaded)',
                {'LIBS':     '-lmkl_gf_lp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp',
                 'LDFLAGS':  '-fopenmp'}],

            # int64_t, threaded
            ['Intel MKL (int64_t, Intel conventions, threaded)',
                {'LIBS':     '-lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp -DMKL_ILP64',
                 'LDFLAGS':  '-fopenmp'}],
            ['Intel MKL (int64_t, GNU conventions, threaded)',
                {'LIBS':     '-lmkl_gf_ilp64 -lmkl_gnu_thread -lmkl_core -lpthread -lm',
                 'CXXFLAGS': '-fopenmp -DMKL_ILP64',
                 'LDFLAGS':  '-fopenmp'}],

            # int, sequential
            ['Intel MKL (int, Intel conventions, sequential)',
                {'LIBS':     '-lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': ''}],
            ['Intel MKL (int, GNU conventions, sequential)',
                {'LIBS':     '-lmkl_gf_lp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': ''}],

            # int64_t, sequential
            ['Intel MKL (int64_t, Intel conventions, sequential)',
                {'LIBS':     '-lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': '-DMKL_ILP64'}],
            ['Intel MKL (int64_t, GNU conventions, sequential)',
                {'LIBS':     '-lmkl_gf_ilp64 -lmkl_sequential -lmkl_core -lm',
                 'CXXFLAGS': '-DMKL_ILP64'}],
        ])
    # end

    if (test_all or test_essl):
        choices.extend([
            ['IBM ESSL', {'LIBS': '-lessl'}],
        ])
    # end

    if (test_all or test_openblas):
        choices.extend([
            ['OpenBLAS', {'LIBS': '-lopenblas'}],
        ])
    # end

    if (test_all or test_accelerate):
        path = '/System/Library/Frameworks/Accelerate.framework/Frameworks/vecLib.framework/Headers'
        inc = '-I' + path if (os.path.exists( path )) else ''
        choices.extend([
            ['MacOS Accelerate',
                {'LIBS': '-framework Accelerate', 'CXXFLAGS': inc}],
        ])
    # end

    #--------------------
    # Deprecated libraries last.
    if (test_all or test_acml):
        choices.extend([
            ['AMD ACML (threaded)', {'LIBS': '-lacml_mp'}],
            ['AMD ACML (sequential)', {'LIBS': '-lacml'}],
        ])
    # end

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
                        print( ansi_red +
                               'Changing Fortran name mangling for both BLAS and LAPACK to '
                               + ' '.join( new_mangling_sizes ) + ansi_normal )
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
        print( ansi_bold + ansi_red + 'unexpected error!' + ansi_normal )
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
        print( ansi_bold + ansi_red + 'unexpected error!' + ansi_normal )
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
