#!/usr/bin/env python
#
# Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.
#
# Usage: python configure.py [--interactive]

from __future__ import print_function

import sys
import re
import config
from   config import Error, font, print_warn
import config.lapack

#-------------------------------------------------------------------------------
# header

print( '-'*80 + '\n' +
font.bold( font.blue( '                              Welcome to BLAS++.' ) ) +
'''

By default, configure will automatically choose the first valid value it finds
for each option. You can set it to interactive to find all possible values and
give you a choice:
    ''' + font.blue( 'make config interactive=1' ) + '''

If you have multiple compilers, we suggest specifying your desired compiler by
setting CXX, as the automated search may prefer a different compiler.

For options, see the `INSTALL.md` file.

Configure assumes environment variables CPATH, LIBRARY_PATH, and LD_LIBRARY_PATH
are set so your compiler can find libraries. See INSTALL.md for more details.
''' + '-'*80 )

#-------------------------------------------------------------------------------
def main():
    config.init( namespace='BLAS', prefix='/opt/slate' )
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
        print_warn( 'BLAS++ needs CBLAS only in testers.' )

    try:
        config.lapack.lapack()
    except Error:
        print_warn( 'BLAS++ needs LAPACK only in testers.' )

    try:
        config.cublas_library()
        config.environ.merge({'devtarget': 'cuda'})
    except Error:
        print_warn( 'BLAS++ CUDA wrappers will not be compiled.' )

    testsweeper = config.get_package(
        'testsweeper',
        ['../testsweeper', './testsweeper'],
        'https://bitbucket.org/icl/testsweeper',
        'https://bitbucket.org/icl/testsweeper/get/master.tar.gz',
        'testsweeper.tar.gz' )
    if (not testsweeper):
        print_warn( 'BLAS++ needs TestSweeper only in testers.' )

    config.extract_defines_from_flags( 'CXXFLAGS', 'blaspp_header_defines' )
    config.output_files( ['make.inc', 'include/blas/defines.h'] )
    print( 'log in config/log.txt' )

    print( '-'*80 )
# end

#-------------------------------------------------------------------------------
try:
    main()
except Error as ex:
    print_warn( 'A fatal error occurred. ' + str(ex) +
                '\nBLAS++ could not be configured. Log in config/log.txt' )
    exit(1)
