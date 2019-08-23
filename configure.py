#!/usr/bin/env python
#
# Usage: python configure.py [--interactive]

from __future__ import print_function

import sys
import re
import config
from   config import ansi_bold, ansi_red, ansi_blue, ansi_normal
from   config import Error
import config.lapack
import os

#-------------------------------------------------------------------------------
# header

print( '-'*80 + '\n' +
ansi_bold + ansi_red + '                              Welcome to BLAS++.' +
ansi_normal + '''

By default, configure will automatically choose the first valid value it finds
for each option. You can set it to interactive to find all possible values and
give you a choice:
    ''' + ansi_blue + 'make config interactive=1' + ansi_normal + '''

If you have multiple compilers, we suggest specifying your desired compiler by
setting CXX, as the automated search may prefer a different compiler. To limit
which versions of BLAS and LAPACK to search for, set one or more of:
    mkl=1, acml=1, essl=1, openblas=1, accelerate=1;
    lp64=1, ilp64=1.
For instance,
    ''' + ansi_blue + 'make config CXX=xlc++ essl=1' + ansi_normal + '''

BLAS and LAPACK are written in Fortran, which has a compiler-specific name
mangling scheme: routine DGEMM is called dgemm_, dgemm, or DGEMM in the
library. (Some libraries like MKL and ESSL support multiple schemes.)
Configure will auto-detect the scheme, but you can also specify it by
setting one or more of the corresponding options:
    fortran_add_=1, fortran_lower=1, fortran_upper=1.

Configure assumes environment variables CPATH, LIBRARY_PATH, and LD_LIBRARY_PATH
are set so your compiler can find libraries. See INSTALL.txt for more details.
''' + '-'*80 )

#-------------------------------------------------------------------------------
def main():
    config.init( prefix='/usr/local/blaspp' )
    config.prog_cxx()
    config.prog_cxx_flags([
        '-O2', '-std=c++11', '-MMD',
        '-Wall',
        '-pedantic',
        '-Wshadow',
        '-Wno-unused-local-typedefs',
        '-Wno-unused-function',
        #'-Wmissing-declarations',
        #'-Wconversion',
        #'-Werror',
    ])
    config.openmp()

    config.lapack.blas()
    print()
    config.lapack.blas_float_return()
    config.lapack.blas_complex_return()
    config.lapack.vendor_version()

    # Must test mkl_version before cblas, to define HAVE_MKL.
    try:
        config.lapack.cblas()
    except Error:
        print( ansi_red + 'BLAS++ needs CBLAS only in testers.' + ansi_normal )

    try:
        config.lapack.lapack()
    except Error:
        print( ansi_red + 'LAPACK++ needs LAPACKE only in testers.' + ansi_normal )

    libtest = config.get_package(
        'libtest',
        ['../libtest', './libtest'],
        'https://bitbucket.org/icl/libtest',
        'https://bitbucket.org/icl/libtest/get/tip.tar.gz',
        'libtest.tar.gz' )
    if (not libtest):
        print( ansi_red + 'BLAS++ needs libtest to compile testers.' + ansi_normal )

    config.extract_defines_from_flags( 'CXXFLAGS' )
    config.output_files( ['make.inc', 'blas_defines.h'] )

    print( '-'*80 )
# end

#-------------------------------------------------------------------------------
try:
    main()
except Error as err:
    print( ansi_bold + ansi_red + 'A fatal error occurred. ' + str(err) + '\n'
           'BLAS++ could not be configured.' + ansi_normal )
    exit(1)
