from __future__ import print_function

import os
import shlex
import subprocess
from   subprocess import PIPE
import math
import sys
import time
import re

# ------------------------------------------------------------------------------
# variables to replace instead of appending/prepending
replace_vars = ['CC', 'CXX', 'NVCC', 'FC', 'AR', 'RANLIB', 'prefix']

# ------------------------------------------------------------------------------
# map file extensions to languages
lang_map = {
    '.c':   'CC',

    '.cc':  'CXX',
    '.cxx': 'CXX',
    '.cpp': 'CXX',

    '.cu':  'NVCC',

    '.f':   'FC',
    '.f90': 'FC',
    '.f77': 'FC',
    '.F90': 'FC',
    '.F77': 'FC',
}

# ------------------------------------------------------------------------------
# map languages to compiler flags
flag_map = {
    'CC':   'CFLAGS',
    'CXX':  'CXXFLAGS',
    'NVCC': 'NVCCFLAGS',
    'FC':   'FFLAGS',
}

# ------------------------------------------------------------------------------
def flatten( data, ltypes=(list, tuple) ):
    '''
    Flattens nested list or tuple.
    Ex: flatten( [1, 2, [3, [4, 5], 6]] ) returns [1, 2, 3, 4, 5, 6]

    see http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    '''
    ltype = type(data)
    data = list(data)
    i = 0
    while i < len(data):
        while isinstance(data[i], ltypes):
            if not data[i]:
                data.pop(i)
                i -= 1
                break
            else:
                data[i:i + 1] = data[i]
        i += 1
    return ltype(data)
# end

#-------------------------------------------------------------------------------
def get( dictionary, key ):
    '''
    Returns dictionary[ key ] or ''
    '''
    if (key in dictionary):
        return dictionary[ key ]
    else:
        return ''
# end

# ------------------------------------------------------------------------------
# ANSI codes
ansi_esc     = chr(0x1B) + '['

ansi_red     = ansi_esc + '31m'
ansi_green   = ansi_esc + '32m'
ansi_yellow  = ansi_esc + '33m'
ansi_blue    = ansi_esc + '34m'
ansi_magenta = ansi_esc + '35m'
ansi_cyan    = ansi_esc + '36m'
ansi_white   = ansi_esc + '37m'

ansi_bold    = ansi_esc + '1m'
ansi_normal  = ansi_esc + '0m';

#-------------------------------------------------------------------------------
def print_header( header ):
    print( '\n' + '-'*80 +
           '\n' + ansi_bold + header + ansi_normal, file=log )
    print( '\n' + ansi_bold + header + ansi_normal )
# end

#-------------------------------------------------------------------------------
def print_subhead( subhead ):
    print( '-'*40 + '\n' +
           subhead, file=log )
    print( subhead )
# end

#-------------------------------------------------------------------------------
def print_line( label ):
    if (label):
        print( '-'*20 + '\n' + label, file=log )
        print( '%-72s' % label, end='' )
        sys.stdout.flush()
# end

#-------------------------------------------------------------------------------
def print_result( label, rc, extra='' ):
    if (label):
        if (rc == 0):
            print( ansi_blue + 'yes'  + ansi_normal, extra, file=log )
            print( ansi_blue + ' yes' + ansi_normal, extra )
        else:
            print( ansi_red + 'no'  + ansi_normal, extra, file=log )
            print( ansi_red + ' no' + ansi_normal, extra )
# end

# ------------------------------------------------------------------------------
# Used for all errors.
# Allows Python Exceptions to fall through, giving tracebacks.
class Error( Exception ):
    pass

class Quit( Error ):
    pass

#-------------------------------------------------------------------------------
# Manages stack of environments, which are dictionaries of name=value pairs.
class Environments( object ):
    def __init__( self ):
        self.stack = [ os.environ, {} ]

    def push( self, env=None ):
        self.stack.append( {} )
        if (env):
            self.merge( env )

    def top( self ):
        return self.stack[-1]

    def pop( self ):
        if (len(self.stack) == 2):
            raise Error( "can't pop last 2 environments" )
        return self.stack.pop()

    def __getitem__( self, key ):
        for env in self.stack[::-1]:
            if (key in env):
                return env[ key ]
        return ''

    def __setitem__( self, key, value ):
        self.stack[ -1 ][ key ] = value

    def append( self, key, val ):
        orig = self[ key ]
        if (val):
            if (orig):
                val = orig + ' ' + val
            self[ key ] = val
        return orig

    def prepend( self, key, val ):
        orig = self[ key ]
        if (val):
            if (orig):
                val = val + ' ' + orig
            self[ key ] = val
        return orig

    def merge( self, env ):
        for key in env:
            if (key in replace_vars):
                self[ key ] = env[ key ]
            elif (key == 'LIBS'):
                self.prepend( key, env[ key ] )
            else:
                self.append( key, env[ key ] )
# end

#-------------------------------------------------------------------------------
def choose( choices ):
    n = len( choices )
    if (n == 0):
        print( ansi_bold + ansi_red + 'none found' + ansi_normal )
        raise Error
    elif (n == 1):
        ##print()
        return 0
    else:
        width = int( math.log10( n ) + 1 )
        print()
        for i in xrange( n ):
            print( '[%*d] %s' % (width, i+1, choices[i]) )
        while (True):
            print( 'choose [1-%d] or quit: ' % (n), end='' )
            sys.stdout.flush()
            i = raw_input()
            if (i == 'q' or i == 'quit'):
                raise Quit
            try:
                i = int( i )
            except:
                i = -1
            if (i >= 1 and i <= len( choices )):
                ##print()
                return i-1
        # end
    # end
# end

#-------------------------------------------------------------------------------
def run( cmd, env=None ):
    environ.push( env )

    if (not isinstance( cmd, str )):
        cmd = ' '.join( flatten( cmd ))

    print( '>>>', cmd, file=log )
    cmd_list = shlex.split( cmd )
    try:
        proc = subprocess.Popen( cmd_list, stdout=PIPE, stderr=PIPE )
        (stdout, stderr) = proc.communicate()
        rc = proc.wait()
        log.write( stdout )
        log.write( ansi_red )
        log.write( stderr )
        log.write( ansi_normal )
        print( 'exit status = %d' % rc, file=log )
    except Exception as err:
        rc = -1
        stdout = ''
        stderr = str(err)

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
# Ex:
# compile_obj( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
def compile_obj( src, env=None, label=None ):
    environ.push( env )

    print_line( label )
    (base, ext) = os.path.splitext( src )
    obj      = base + '.o'
    lang     = lang_map[ ext ]
    compiler = environ[ lang ]
    flags    = environ[ flag_map[ lang ]]
    (rc, stdout, stderr) = run([ compiler, flags, '-c', src, '-o', obj ])
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
# Ex:
# link_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
def link_exe( src, env=None, label=None ):
    environ.push( env )

    print_line( label )
    (base, ext) = os.path.splitext( src )
    obj      = base + '.o'
    lang     = lang_map[ ext ]
    compiler = environ[ lang ]
    LDFLAGS  = environ[ 'LDFLAGS' ]
    LIBS     = environ[ 'LIBS' ] or environ[ 'LDLIBS' ]
    (rc, stdout, stderr) = run([ compiler, obj, '-o', base, LDFLAGS, LIBS ])
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
# Ex:
# compile_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
def compile_exe( src, env=None, label=None ):
    environ.push( env )

    print_line( label )
    (base, ext) = os.path.splitext( src )
    obj      = base + '.o'
    lang     = lang_map[ ext ]
    compiler = environ[ lang ]
    LDFLAGS  = environ[ 'LDFLAGS' ]
    LIBS     = environ[ 'LIBS' ] or environ[ 'LDLIBS' ]
    (rc, stdout, stderr) = compile_obj( src )
    if (rc == 0):
        (rc, stdout, stderr) = run([ compiler, obj, '-o', base, LDFLAGS, LIBS ])
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
# Ex:
# compile_run( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
def compile_run( src, env=None, label=None ):
    environ.push( env )

    print_line( label )
    (base, ext) = os.path.splitext( src )
    (rc, stdout, stderr) = compile_exe( src )
    if (rc == 0):
        (rc, stdout, stderr) = run( './' + base )
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
# Ex:
# run_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
def run_exe( src, env=None, label=None ):
    environ.push( env )

    print_line( label )
    (base, ext) = os.path.splitext( src )
    (rc, stdout, stderr) = run( './' + base )
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
def prog_cxx( choices=['g++', 'c++', 'CC', 'cxx', 'icpc', 'xlc++', 'clang++'] ):
    print_header( 'C++ compiler' )

    cxx = environ['CXX']
    if (cxx):
        print( 'Trying $CXX =', cxx )
        choices = [ cxx ]

    passed = []
    for cxx in choices:
        print_line( cxx )
        (rc, out, err) = compile_run( 'config/compiler_cxx.cc', {'CXX': cxx} )
        # print (g++), (clang++), etc., as output by compiler_cxx, after yes
        if (rc == 0):
            out = '(' + out.strip() + ')'
        print_result( cxx, rc, out )
        if (rc == 0):
            passed.append( cxx )
            if (auto): break
        # end
    # end
    i = choose( passed )
    environ['CXX'] = passed[i]
# end

#-------------------------------------------------------------------------------
def prog_cxx_flags( flags ):
    '''
    Tests each flag in flags; if it passes, adds the flag to CXXFLAGS.
    '''
    print_header( 'C++ compiler flags' )
    for flag in flags:
        print_line( flag )
        (rc, out, err) = compile_obj( 'config/compiler_cxx.cc', {'CXXFLAGS': flag} )
        # assume a mention of the flag in stderr means it isn't supported
        if (flag in err):
            rc = 1
        print_result( flag, rc )
        if (rc == 0):
            environ.append( 'CXXFLAGS', flag )
    # end
# end

#-------------------------------------------------------------------------------
def openmp( flags=['-fopenmp', '-qopenmp', '-openmp', '-omp', ''] ):
    '''
    Tests for OpenMP support with one of the given flags.
    '''
    print_header( 'OpenMP support' )
    src = 'config/openmp.cc'
    for flag in flags:
        print_line( flag )
        env = {'CXXFLAGS': flag, 'LDFLAGS': flag}
        (rc, out, err) = compile_run( src, env )
        print_result( flag, rc )
        if (rc == 0):
            environ.merge( env )
            break
    # end
# end

#-------------------------------------------------------------------------------
def sub_env( match ):
    return environ[ match.group(1) ]

#-------------------------------------------------------------------------------
def read( filename ):
    f = open( filename, 'r' )
    txt = f.read()
    f.close()
    return txt
# end

#-------------------------------------------------------------------------------
def write( filename, txt ):
    f = open( filename, 'w' )
    f.write( txt )
    f.close()
# end

#-------------------------------------------------------------------------------
def output_files( files ):
    '''
    Create each file in files from file.in, substituting @foo@ with variable foo.
    This avoids re-creating the file if the contents did not change.
    files can be a single file or list of files.
    '''
    print_header( 'Output files' )
    if (isinstance( files, str )):
        files = [ files ]
    for fname in files:
        txt = read( fname + '.in' )
        out = re.sub( r'@(\w+)@', sub_env, txt )
        exists = os.path.exists( fname )
        if (exists and out == read( fname )):
            print( fname, 'is unchanged' )
        else:
            if (exists):
                bak = fname + '.bak'
                print( 'backing up', fname, 'to', bak )
                os.rename( fname, bak )
            # end
            print( 'creating', fname )
            write( fname, out )
        # end
    # end
# end

#-------------------------------------------------------------------------------
def init( prefix='/usr/local' ):
    global environ, log, auto

    environ['prefix'] = prefix

    #--------------------
    logfile = 'config/log.txt'
    print( 'opening log file ' + logfile + '\n' )
    log = open( logfile, 'w' )

    #--------------------
    # Workaround if MacOS SIP may have prevented inheriting DYLD_LIBRARY_PATH.
    if (sys.platform.startswith('darwin') and
        'LD_LIBRARY_PATH' not in os.environ and
        'DYLD_LIBRARY_PATH' not in os.environ):
        print( ansi_bold + ansi_red +
               'NOTICE: $DYLD_LIBRARY_PATH was not inherited (or not set).' )
        if ('LIBRARY_PATH' in os.environ):
            print( 'Setting $DYLD_LIBRARY_PATH = $LIBRARY_PATH to run test programs.' )
            os.environ['DYLD_LIBRARY_PATH'] = os.environ['LIBRARY_PATH']
            print( ansi_red + 'set $DYLD_LIBRARY_PATH = $LIBRARY_PATH =',
                   os.environ['LIBRARY_PATH'] + ansi_normal, file=log )
        else:
            print( '$LIBRARY_PATH is also not set. Leaving $DYLD_LIBRARY_PATH unset.' )
            print( ansi_red +
                   '$LIBRARY_PATH is also not set. Leaving $DYLD_LIBRARY_PATH unset.'
                   + ansi_normal,
                   file=log )
        # end
        print( ansi_normal + ansi_red + '''\
MacOS System Integrity Protection (SIP) prevents configure.py from inheriting
$DYLD_LIBRARY_PATH. Using
    python configure.py
directly (not via make), with python installed from python.org (not Apple's
python in /usr/bin), will allow $DYLD_LIBRARY_PATH to be inherited.'''
+ ansi_normal )
    # end

    #--------------------
    # parse command line
    for arg in sys.argv[1:]:
        if (arg == '--interactive' or arg == '-i'):
            auto = False
        else:
            s = re.search( '^(\w+)=(.*)', arg )
            if (s):
                environ[ s.group(1) ] = s.group(2)
            else:
                print( 'Unknown argument:', arg )
                exit(1)
    # end

    if (environ['interactive'] == '1'):
        auto = False
# end

# ------------------------------------------------------------------------------
# Initialize global variables here, rather than in init(),
# so they are exported to __init__.py.
environ = Environments()
environ['argv'] = ' '.join( sys.argv )
environ['datetime'] = time.ctime()

auto = True
