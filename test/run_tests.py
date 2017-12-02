#!/usr/bin/env python
#
# Example usage:
# help
#     ./run_tests.py -h
#
# run everything with default sizes
# output is redirected; summary information is printed on stderr
#     ./run_tests.py > output.txt
#
# run Level 1 BLAS (axpy, dot, ...)
# with float, double and default sizes
#     ./run_tests.py -f -d --blas1
#
# run gemm, gemv with small, medium sizes
#     ./run_tests.py -s -m gemm gemv

from __future__ import print_function

import sys
import os
import re
import argparse

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_type = parser.add_argument_group( 'precisions (default is all)' )
group_type.add_argument( '-f', '--float',          action='store_true', help='run float (single precision) tests' )
group_type.add_argument( '-d', '--double',         action='store_true', help='run double precision tests' )
group_type.add_argument( '-c', '--complex-float',  action='store_true', help='run complex-float precision tests' )
group_type.add_argument( '-z', '--complex-double', action='store_true', help='run complex-double precision tests' )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '-x', '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '-l', '--large',  action='store_true', help='run large tests' )
group_size.add_argument(       '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    group_cat.add_argument( '--blas1', action='store_true', help='run Level 1 BLAS tests' ),
    group_cat.add_argument( '--blas2', action='store_true', help='run Level 2 BLAS tests' ),
    group_cat.add_argument( '--blas3', action='store_true', help='run Level 3 BLAS tests' ),
]
categories = map( lambda x: x.dest, categories ) # map to names: ['lu', 'chol', ...]

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

# by default, run all precisions
if (not (opts.float or opts.double or opts.complex_float or opts.complex_double)):
    opts.float          = True
    opts.double         = True
    opts.complex_float  = True
    opts.complex_double = True

# by default, run medium sizes
if (not (opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all categories
if (opts.tests or not any( map( lambda c: opts.__dict__[ c ], categories ))):
    for c in categories:
        opts.__dict__[ c ] = True

# ------------------------------------------------------------------------------
# parameters
# begin with space to ease concatenation

# if given, use explicit dim
dim = ' --dim ' + opts.dim if (opts.dim) else ''
n        = dim
tall     = dim
wide     = dim
mn       = dim
mnk      = dim
nk_tall  = dim
nk_wide  = dim
nk       = dim

if (not opts.dim):
    if (opts.xsmall):
        n       += ' --dim 10'
        tall    += ' --dim 20x10'
        wide    += ' --dim 10x20'
        mnk     += ' --dim 10x15x20 --dim 15x10x20' \
                +  ' --dim 10x20x15 --dim 15x20x10' \
                +  ' --dim 20x10x15 --dim 20x15x10'
        nk_tall += ' --dim 1x20x10'
        nk_wide += ' --dim 1x10x20'

    if (opts.small):
        n       += ' --dim 25:100:25'
        tall    += ' --dim 50:200:50x25:100:25'  # 2:1
        wide    += ' --dim 25:100:25x50:200:50'  # 1:2
        mnk     += ' --dim 25x50x75 --dim 50x25x75' \
                +  ' --dim 25x75x50 --dim 50x75x25' \
                +  ' --dim 75x25x50 --dim 75x50x25'
        nk_tall += ' --dim 1x50:200:50x25:100:25'
        nk_wide += ' --dim 1x25:100:25x50:200:50'

    if (opts.medium):
        n       += ' --dim 100:500:100'
        tall    += ' --dim 200:1000:200x100:500:100'  # 2:1
        wide    += ' --dim 100:500:100x200:1000:200'  # 1:2
        mnk     += ' --dim 100x300x600 --dim 300x100x600' \
                +  ' --dim 100x600x300 --dim 300x600x100' \
                +  ' --dim 600x100x300 --dim 600x300x100'
        nk_tall += ' --dim 1x200:1000:200x100:500:100'
        nk_wide += ' --dim 1x100:500:100x200:1000:200'

    if (opts.large):
        n       += ' --dim 1000:5000:1000'
        tall    += ' --dim 2000:10000:2000x1000:5000:1000'  # 2:1
        wide    += ' --dim 1000:5000:1000x2000:10000:2000'  # 1:2
        mnk     += ' --dim 1000x3000x6000 --dim 3000x1000x6000' \
                +  ' --dim 1000x6000x3000 --dim 3000x6000x1000' \
                +  ' --dim 6000x1000x3000 --dim 6000x3000x1000'
        nk_tall += ' --dim 1x2000:10000:2000x1000:5000:1000'
        nk_wide += ' --dim 1x1000:5000:1000x2000:10000:2000'

    mn  = n + tall + wide
    mnk = mn + mnk
    nk  = n + nk_tall + nk_wide
# end

incx_pos = ' --incx 1,2'
incx     = ' --incx 1,2,-1,-2'
incy_pos = ' --incy 1,2'
incy     = ' --incy 1,2,-1,-2'

dtypes = []
if (opts.float):  dtypes.append( 's' )
if (opts.double): dtypes.append( 'd' )
if (opts.complex_float):  dtypes.append( 'c' )
if (opts.complex_double): dtypes.append( 'z' )
dtype         = ' --type ' + ','.join( dtypes )

r = filter( lambda x: x in ('s', 'd'), dtypes )
if (r):
    dtype_real = ' --type ' + ','.join( r )
else:
    dtype_real = ''

c = filter( lambda x: x in ('c', 'z'), dtypes )
if (c):
    dtype_complex = ' --type ' + ','.join( c )
else:
    dtype_complex = ''

layout   = ' --layout c,r'
transA   = ' --transA n,t,c'
transB   = ' --transB n,t,c'
trans    = ' --trans n,t,c'
trans_nt = ' --trans n,t'
trans_nc = ' --trans n,c'
uplo     = ' --uplo l,u'
norm     = ' --norm 1,inf,fro,max'
diag     = ' --diag n,u'
direct   = ' --direct f,b'
storev   = ' --storev c,r'
side     = ' --side l,r'
mtype    = ' --matrixtype g,l,u'
align    = ' --align 32'

# ------------------------------------------------------------------------------
cmds = []

# Level 1
if (opts.blas1):
    cmds += [
    [ 'asum',  dtype      + n + incx_pos ],
    [ 'axpy',  dtype      + n + incx + incy ],
    [ 'copy',  dtype      + n + incx + incy ],
    [ 'dot',   dtype      + n + incx + incy ],
    [ 'dotu',  dtype      + n + incx + incy ],
    [ 'iamax', dtype      + n + incx_pos ],
    [ 'nrm2',  dtype      + n + incx_pos ],
    [ 'rot',   dtype_real + n + incx + incy ],
    [ 'rotm',  dtype_real + n + incx + incy ],
    [ 'scal',  dtype      + n + incx_pos ],
    [ 'swap',  dtype      + n + incx + incy ],
    ]

# Level 2
if (opts.blas2):
    cmds += [
    [ 'gemv',  dtype      + layout + align + trans + mn + incx + incy ],
    [ 'ger',   dtype      + layout + align + mn + incx + incy ],
    [ 'geru',  dtype      + layout + align + mn + incx + incy ],
    [ 'hemv',  dtype      + layout + align + uplo + n + incx + incy ],
    [ 'her',   dtype      + layout + align + uplo + n + incx ],
    [ 'her2',  dtype      + layout + align + uplo + n + incx + incy ],
    [ 'symv',  dtype      + layout + align + uplo + n + incx + incy ],
    [ 'syr',   dtype_real + layout + align + uplo + n + incx ], # complex is in lapack++
    [ 'syr2',  dtype      + layout + align + uplo + n + incx + incy ],
    [ 'trmv',  dtype      + layout + align + uplo + trans + diag + n + incx ],
    [ 'trsv',  dtype      + layout + align + uplo + trans + diag + n + incx ],
    ]

# Level 3
if (opts.blas3):
    cmds += [
    [ 'gemm',  dtype         + layout + align + transA + transB + mnk ],
    [ 'hemm',  dtype         + layout + align + side + uplo + mn ],
    [ 'symm',  dtype         + layout + align + side + uplo + mn ],
    [ 'trmm',  dtype         + layout + align + side + uplo + trans + diag + mn ],
    [ 'trsm',  dtype         + layout + align + side + uplo + trans + diag + mn ],
    [ 'herk',  dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'herk',  dtype_complex + layout + align + uplo + trans_nc + mn ],
    [ 'syrk',  dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'syrk',  dtype_complex + layout + align + uplo + trans_nt + mn ],
    [ 'her2k', dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'her2k', dtype_complex + layout + align + uplo + trans_nc + mn ],
    [ 'syr2k', dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'syr2k', dtype_complex + layout + align + uplo + trans_nt + mn ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
def run_test( cmd ):
    cmd = './test %-6s%s' % tuple(cmd)
    print( cmd, file=sys.stderr )
    err = os.system( cmd )
    if (err):
        hi = (err & 0xff00) >> 8
        lo = (err & 0x00ff)
        if (lo == 2):
            print( '\nCancelled', file=sys.stderr )
            exit(1)
        elif (lo != 0):
            print( 'FAILED: abnormal exit, signal =', lo, file=sys.stderr )
        elif (output_redirected):
            print( hi, 'tests FAILED.', file=sys.stderr )
    # end
    return err
# end

# ------------------------------------------------------------------------------
failures = []
run_all = (len(opts.tests) == 0)
for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        err = run_test( cmd )
        if (err != 0):
            failures.append( cmd[0] )

# print summary of failures
nfailures = len( failures )
if (nfailures > 0):
    print( '\n' + str(nfailures) + ' routines FAILED:', ', '.join( failures ),
           file=sys.stderr )
