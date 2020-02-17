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
# with single, double and default sizes
#     ./run_tests.py --blas1 --type s,d
#
# run gemm, gemv with small, medium sizes
#     ./run_tests.py -s -m gemm gemv

from __future__ import print_function

import sys
import os
import re
import argparse
import subprocess
import xml.etree.ElementTree as ET
import io

# ------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()

group_test = parser.add_argument_group( 'test' )
group_test.add_argument( '-t', '--test', action='store',
    help='test command to run, e.g., --test "mpirun -np 4 ./test"; default "%(default)s"',
    default='./tester' )
group_test.add_argument( '--xml', help='generate report.xml for jenkins' )

group_size = parser.add_argument_group( 'matrix dimensions (default is medium)' )
group_size.add_argument( '-x', '--xsmall', action='store_true', help='run x-small tests' )
group_size.add_argument( '-s', '--small',  action='store_true', help='run small tests' )
group_size.add_argument( '-m', '--medium', action='store_true', help='run medium tests' )
group_size.add_argument( '-l', '--large',  action='store_true', help='run large tests' )
group_size.add_argument(       '--square', action='store_true', help='run square (m = n = k) tests', default=False )
group_size.add_argument(       '--tall',   action='store_true', help='run tall (m > n) tests', default=False )
group_size.add_argument(       '--wide',   action='store_true', help='run wide (m < n) tests', default=False )
group_size.add_argument(       '--mnk',    action='store_true', help='run tests with m, n, k all different', default=False )
group_size.add_argument(       '--dim',    action='store',      help='explicitly specify size', default='' )

group_cat = parser.add_argument_group( 'category (default is all)' )
categories = [
    group_cat.add_argument( '--blas1', action='store_true', help='run Level 1 BLAS tests' ),
    group_cat.add_argument( '--blas2', action='store_true', help='run Level 2 BLAS tests' ),
    group_cat.add_argument( '--blas3', action='store_true', help='run Level 3 BLAS tests' ),
    group_cat.add_argument( '--batch-blas3', action='store_true', help='run Level 3 Batch BLAS tests' ),
]
# map category objects to category names: ['lu', 'chol', ...]
categories = list( map( lambda x: x.dest, categories ) )

group_opt = parser.add_argument_group( 'options' )
# BLAS and LAPACK
# Empty defaults (check, ref, etc.) use the default in test.cc.
group_opt.add_argument( '--type',   action='store', help='default=%(default)s', default='s,d,c,z' )
group_opt.add_argument( '--layout', action='store', help='default=%(default)s', default='c,r' )
group_opt.add_argument( '--transA', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--transB', action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--trans',  action='store', help='default=%(default)s', default='n,t,c' )
group_opt.add_argument( '--uplo',   action='store', help='default=%(default)s', default='l,u' )
group_opt.add_argument( '--diag',   action='store', help='default=%(default)s', default='n,u' )
group_opt.add_argument( '--side',   action='store', help='default=%(default)s', default='l,r' )
group_opt.add_argument( '--alpha',  action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--beta',   action='store', help='default=%(default)s', default='' )
group_opt.add_argument( '--incx',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--incy',   action='store', help='default=%(default)s', default='1,2,-1,-2' )
group_opt.add_argument( '--align',  action='store', help='default=%(default)s', default='32' )
group_opt.add_argument( '--check',  action='store', help='default=y', default='' )  # default in test.cc
group_opt.add_argument( '--ref',    action='store', help='default=y', default='' )  # default in test.cc

parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

for t in opts.tests:
    if (t.startswith('--')):
        print( 'Error: option', t, 'must come before any routine names' )
        print( 'usage:', sys.argv[0], '[options]', '[routines]' )
        print( '      ', sys.argv[0], '--help' )
        exit(1)

# by default, run medium sizes
if (not (opts.xsmall or opts.small or opts.medium or opts.large)):
    opts.medium = True

# by default, run all shapes
if (not (opts.square or opts.tall or opts.wide or opts.mnk)):
    opts.square = True
    opts.tall   = True
    opts.wide   = True
    opts.mnk    = True

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

    mn  = ''
    nk  = ''
    if (opts.square):
        mn = n
        nk = n
    if (opts.tall):
        mn += tall
        nk += nk_tall
    if (opts.wide):
        mn += wide
        nk += nk_wide
    if (opts.mnk):
        mnk = mn + mnk
    else:
        mnk = mn
# end

# BLAS and LAPACK
dtype  = ' --type '   + opts.type   if (opts.type)   else ''
layout = ' --layout ' + opts.layout if (opts.layout) else ''
transA = ' --transA ' + opts.transA if (opts.transA) else ''
transB = ' --transB ' + opts.transB if (opts.transB) else ''
trans  = ' --trans '  + opts.trans  if (opts.trans)  else ''
uplo   = ' --uplo '   + opts.uplo   if (opts.uplo)   else ''
diag   = ' --diag '   + opts.diag   if (opts.diag)   else ''
side   = ' --side '   + opts.side   if (opts.side)   else ''
a      = ' --alpha '  + opts.alpha  if (opts.alpha)  else ''
ab     = a+' --beta ' + opts.beta   if (opts.beta)   else a
incx   = ' --incx '   + opts.incx   if (opts.incx)   else ''
incy   = ' --incy '   + opts.incy   if (opts.incy)   else ''
align  = ' --align '  + opts.align  if (opts.align)  else ''
check  = ' --check '  + opts.check  if (opts.check)  else ''
ref    = ' --ref '    + opts.ref    if (opts.ref)    else ''

# ------------------------------------------------------------------------------
# filters a comma separated list csv based on items in list values.
# if no items from csv are in values, returns first item in values.
def filter_csv( values, csv ):
    f = list( filter( lambda x: x in values, csv.split( ',' ) ) )
    if (not f):
        return values[0]
    return ','.join( f )
# end

# ------------------------------------------------------------------------------
# limit options to specific values
dtype_real    = ' --type ' + filter_csv( ('s', 'd'), opts.type )
dtype_complex = ' --type ' + filter_csv( ('c', 'z'), opts.type )
dtype_double  = ' --type ' + filter_csv( ('d', 'z'), opts.type )

trans_nt = ' --trans ' + filter_csv( ('n', 't'), opts.trans )
trans_nc = ' --trans ' + filter_csv( ('n', 'c'), opts.trans )

# positive inc
incx_pos = ' --incx ' + filter_csv( ('1', '2'), opts.incx )
incy_pos = ' --incy ' + filter_csv( ('1', '2'), opts.incy )

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
    [ 'symv',  dtype_real + layout + align + uplo + n + incx + incy ], # complex is in lapack++
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

# Batch Level 3
if (opts.batch_blas3):
    cmds += [
    [ 'batch-gemm',  dtype         + layout + align + transA + transB + mnk ],
    [ 'batch-hemm',  dtype         + layout + align + side + uplo + mn ],
    [ 'batch-symm',  dtype         + layout + align + side + uplo + mn ],
    [ 'batch-trmm',  dtype         + layout + align + side + uplo + trans + diag + mn ],
    [ 'batch-trsm',  dtype         + layout + align + side + uplo + trans + diag + mn ],
    [ 'batch-herk',  dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'batch-herk',  dtype_complex + layout + align + uplo + trans_nc + mn ],
    [ 'batch-syrk',  dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'batch-syrk',  dtype_complex + layout + align + uplo + trans_nt + mn ],
    [ 'batch-her2k', dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'batch-her2k', dtype_complex + layout + align + uplo + trans_nc + mn ],
    [ 'batch-syr2k', dtype_real    + layout + align + uplo + trans    + mn ],
    [ 'batch-syr2k', dtype_complex + layout + align + uplo + trans_nt + mn ],
    ]

# ------------------------------------------------------------------------------
# when output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
output_redirected = not sys.stdout.isatty()

# ------------------------------------------------------------------------------
# if output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

# ------------------------------------------------------------------------------
# cmd is a pair of strings: (function, args)

def run_test( cmd ):
    cmd = opts.test +' '+ cmd[0] +' '+ cmd[1]
    print_tee( cmd )
    output = ''
    p = subprocess.Popen( cmd.split(), stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT )
    p_out = p.stdout
    if (sys.version_info.major >= 3):
        p_out = io.TextIOWrapper(p.stdout, encoding='utf-8')
    # Read unbuffered ("for line in p.stdout" will buffer).
    for line in iter(p_out.readline, ''):
        print( line, end='' )
        output += line
    err = p.wait()
    if (err != 0):
        print_tee( 'FAILED: exit code', err )
    else:
        print_tee( 'pass' )
    return (err, output)
# end

# ------------------------------------------------------------------------------
# run each test
failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

seen = set()
for cmd in cmds:
    if (run_all or cmd[0] in opts.tests):
        seen.add( cmd[0] )
        (err, output) = run_test( cmd )
        if (err):
            failed_tests.append( (cmd[0], err, output) )
        else:
            passed_tests.append( cmd[0] )
not_seen = list( filter( lambda x: x not in seen, opts.tests ) )

if (not_seen):
    print_tee( 'Warning: unknown routines:', ' '.join( not_seen ))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( '\n' + str(nfailed) + ' routines FAILED:',
               ', '.join( [x[0] for x in failed_tests] ) )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="blaspp_suite",
                        tests=str(ntests),
                        errors="0",
                        failures=str(nfailed))

    for (test, err, output) in failed_tests:
        testcase = ET.SubElement(doc, "testcase", name=test)

        failure = ET.SubElement(testcase, "failure")
        if (err < 0):
            failure.text = "exit with signal " + str(-err)
        else:
            failure.text = str(err) + " tests failed"

        system_out = ET.SubElement(testcase, "system-out")
        system_out.text = output
    # end

    for test in passed_tests:
        testcase = ET.SubElement(doc, 'testcase', name=test)
        testcase.text = 'PASSED'

    tree = ET.ElementTree(root)
    tree.write( opts.xml )
# end

exit( nfailed )
