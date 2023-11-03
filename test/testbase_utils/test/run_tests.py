#!/usr/bin/env python3
#
# Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

from __future__ import print_function

import sys
import os
import re
import argparse
import subprocess
import xml.etree.ElementTree as ET
import io
import time

#-------------------------------------------------------------------------------
# command line arguments
parser = argparse.ArgumentParser()
parser.add_argument( '--xml', help='generate report.xml for jenkins' )
parser.add_argument( 'tests', nargs=argparse.REMAINDER )
opts = parser.parse_args()

opts.tests = list( map( int, opts.tests ) )

#-------------------------------------------------------------------------------
# 4 tuple: [ index, command, expected exit code=0 ]
cmds = [
    #----------
    # Basics
    #
    # help (no input)
    [ 0, './tester' ],

    # help -h
    [ 1, './tester -h' ],

    # help --help
    [ 2, './tester --help' ],

    # routine help -h
    [ 3, './tester -h sort' ],

    # routine help --help
    [ 4, './tester --help sort' ],

    # Defaults (--type d --dim 100:500:100).
    [ 5, './tester sort' ],

    # Larger range; should elicit 2 failures (error = 1.23456e-17 * n).
    [ 6, './tester --dim 100:1000:100 sort', 2 ],

    #----------
    # Types (enum)
    #
    # Specify types.
    [ 100, './tester --type s,d sort' ],

    # Invalid type x; should return error.
    [ 101, './tester --type s,x,d sort', 255 ],

    #----------
    # Dimensions
    #
    # m == n == k
    [ 200, './tester --type s --dim 100:300:100 sort2' ],

    # m == n == k, descending
    [ 201, './tester --type s --dim 300:100:-100 sort2' ],

    # single dimension
    [ 202, './tester --type s --dim 1234 sort2' ],

    # multiple --dim
    [ 203, './tester --type s --dim 1234 --dim 100:300:100 sort2' ],

    # metric and binary prefix
    [ 204, './tester --dim 1k:4kx1ki:4ki --dim 1M:4Mx1Mi:4Mi --dim 1G:4Gx1Gi:4Gi --dim 1T:4Tx1Ti:4Ti --dim 1P:4Px1Pi:4Pi --dim 1E:4Ex1Ei:4Ei sort2', 24 ],

    # exponent
    [ 205, './tester --dim 1e3:4e3 --dim 1e6:4e6 sort2', 8 ],

    # single nb
    [ 206, './tester --nb 32 --dim 100 sort2' ],

    # range nb
    [ 207, './tester --nb 32:256:32 --dim 100 sort2' ],

    # with illegal step = start = 0
    [ 208, './tester --nb 0:5 sort2', 255 ],

    #----------
    # Zip of dimensions
    #
    # m x n == k.
    [ 300, './tester --type s --dim 100:300:100x50:200:50 sort3' ],

    # m x n; k fixed.
    [ 301, './tester --type s --dim 100:300:100x50:200:50x50 sort3' ],

    # m; n, k fixed.
    [ 302, './tester --type s --dim 100:300:100x100x50 sort3' ],

    # m x n x k.
    [ 303, './tester --type s --dim 100:300:100x50:200:50x10:50:10 sort3' ],

    #----------
    # Cartesian product of dimensions
    #
    # m * n == k
    [ 400, './tester --type s --dim 100:300:100*50:200:50 sort4' ],

    # m * n * k
    [ 401, './tester --type s --dim 100:300:100*50:200:50*10:50:10 sort4' ],

    # m fixed * n * k
    [ 402, './tester --type s --dim 100*50:200:50*10:50:10 sort4' ],

    # m * n fixed * k
    [ 403, './tester --type s --dim 100:300:100*50*10:50:10 sort4' ],

    # m * n * k fixed
    [ 404, './tester --type s --dim 100:300:100*50:200:50*10 sort4' ],

    #----------
    # Check and ref
    #
    # check y
    [ 500, './tester --check y sort5' ],

    # check n
    [ 501, './tester --check n sort5' ],

    # ref y
    [ 502, './tester --ref y sort5' ],

    # ref n
    [ 503, './tester --ref n sort5' ],

    #----------
    # Float and complex
    [ 600, './tester --alpha -2,0,2 sort6' ],

    # inf
    [ 601, './tester --alpha -inf,0,inf sort6', 10 ],

    # complex
    [ 602, './tester --alpha 1.23+2.34i,1.23-2.34i --dim 100 sort6' ],

    # float range
    [ 603, './tester --beta 1.234:5.678:0.5 --dim 100 sort6' ],

    # with step = start
    [ 604, './tester --beta 2.5:12.5 --dim 100 sort6' ],

    # with illegal step = start = 0
    [ 605, './tester --beta 0:12.5 --dim 100 sort6', 255 ],

    # with start = 0, step != 0
    [ 606, './tester --beta 0:12.5:1.25 --dim 100 sort6' ],
]

#-------------------------------------------------------------------------------
# When output is redirected to file instead of TTY console,
# print extra messages to stderr on TTY console.
#
output_redirected = not sys.stdout.isatty()

#-------------------------------------------------------------------------------
# If output is redirected, prints to both stderr and stdout;
# otherwise prints to just stdout.
#
def print_tee( *args ):
    global output_redirected
    print( *args )
    if (output_redirected):
        print( *args, file=sys.stderr )
# end

#-------------------------------------------------------------------------------
# Used in
#     re.sub( pattern, strip_time_sub, line )
# where pattern matches 4 time or Gflop/s fields to replace with hyphens,
# and status field to keep.
# If re.sub matched line '    3.5       4.5  12.34   5.23  pass',
# this returns string    '  -----  --------  -----  -----  pass'
#
def strip_time_sub( match ):
    result = ''
    for i in range( 1, 5 ):
        result += '  ' + '-' * (len( match.group( i ) ) - 2)
    result += '  ' + match.group( 5 )
    return result
# end

#-------------------------------------------------------------------------------
# Runs cmd. Returns exit code and output (stdout and stderr merged).
#
def run_test( num, cmd, expected_err=0 ):
    print_tee( str(num) + ': ' + cmd )
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
    if (err == expected_err):
        if (err == 0):
            print_tee( 'pass' )
        else:
            print_tee( 'pass: got expected exit code =', err )
    else:
        print_tee( 'FAILED: got exit code = %d, expected exit code = %d'
                   % (err, expected_err) )
    # end

    # Save output.
    outfile = 'out/%03d.txt' % (num)
    print( 'Saving to', outfile )
    try:
        # Always strip out ANSI codes and version numbers.
        output2 = re.sub( r'\x1B\[\d+m', r'', output )
        output2 = re.sub( r'version \S+, id \S+',
                          r'version NA, id NA', output2 )
        # Strip out 4 time and Gflop/s fields before status.
        # Using ( +(?:\d+\.\d+|inf|NA)){4} captures only 1 group, the last,
        # hence repeating it 4 times to capture 4 groups.
        output2 = re.sub(
            r'( +(?:\d+\.\d+|inf|NA))( +(?:\d+\.\d+|inf|NA))( +(?:\d+\.\d+|inf|NA))( +(?:\d+\.\d+|inf|NA)) +(pass|FAIL|no check)',
            strip_time_sub, output2 )
        # end
        out = open( outfile, 'w' )
        out.write( output2 )
        out.close()
    except Exception as ex:
        print_tee( 'FAILED: ' + outfile + ': ' + str(ex) )
        err = -2

    # Compare with reference.
    reffile = 'ref/%03d.txt' % (num)
    print( 'Comparing to', reffile )
    try:
        file = open( reffile )
        ref = file.read()
        if (ref != output2):
            print_tee( 'FAILED: diff', outfile, reffile )
            err = -1
    except Exception as ex:
        print_tee( 'FAILED: ' + reffile + ': ' + str(ex) )
        err = -2

    return (err, output)
# end

#-------------------------------------------------------------------------------
# Utility to pretty print XML.
# See https://stackoverflow.com/a/33956544/1655607
#
def indent_xml( elem, level=0 ):
    i = "\n" + level*"  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml( elem, level+1 )
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i
# end

#-------------------------------------------------------------------------------
# run each test

start = time.time()
print_tee( time.ctime() )

failed_tests = []
passed_tests = []
ntests = len(opts.tests)
run_all = (ntests == 0)

if (not os.path.exists( 'out' )):
    os.mkdir( 'out' )

seen = set()
for tst in cmds:
    num = tst[0]
    cmd = tst[1]

    expected_err = 0
    if (len( tst ) > 2):
        expected_err = tst[2]

    if (run_all or tst[0] in opts.tests):
        seen.add( tst[0] )
        (err, output) = run_test( num, cmd, expected_err )
        if (err != expected_err):
            failed_tests.append( (cmd, err, output) )
        else:
            passed_tests.append( cmd )

not_seen = list( filter( lambda x: x not in seen, opts.tests ) )
if (not_seen):
    print_tee( 'Warning: unknown tests:', ' '.join( map( str, not_seen )))

# print summary of failures
nfailed = len( failed_tests )
if (nfailed > 0):
    print_tee( '\n' + str(nfailed) + ' tests FAILED:\n' +
               '\n'.join( [x[0] for x in failed_tests] ) )
else:
    print_tee( '\n' + 'All tests passed.' )

# generate jUnit compatible test report
if opts.xml:
    print( 'writing XML file', opts.xml )
    root = ET.Element("testsuites")
    doc = ET.SubElement(root, "testsuite",
                        name="TestSweeper_suite",
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
    indent_xml( root )
    tree.write( opts.xml )
# end

elapsed = time.time() - start
print_tee( 'Elapsed %.2f sec' % elapsed )
print_tee( time.ctime() )

exit( nfailed )
