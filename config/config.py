from __future__ import print_function

import os
import shlex
import subprocess
from   subprocess import PIPE
import math
import sys
import time
import re
import tarfile
import urllib

# Python 3 renames raw_input => input.
if (sys.version_info.major < 3):
    input = raw_input

#-------------------------------------------------------------------------------
def urlretrieve( url, filename ):
    '''
    Downloads url and saves to filename.
    Works for both Python 2 and 3, which differ in where urlretrieve is located.
    '''
    if (sys.version_info.major >= 3):
        urllib.requests.urlretrieve( url, filename )
    else:
        urllib.urlretrieve( url, filename )
# end

#-------------------------------------------------------------------------------
interactive_ = False

# Function to get and set interactive flag. If True, config finds all possible
# values and gives user a choice. If False, config picks the first valid value.
# value = interactive() returns value of interactive.
# interactive( value ) sets value of interactive.
def interactive( value=None ):
    global interactive_
    if (value is not None):
        interactive_ = value
    return interactive_
# end

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
    '''
    Prints a header, with bold font, both to console and the log.
    '''
    print( '\n' + '-'*80 +
           '\n' + ansi_bold + header + ansi_normal, file=log )
    print( '\n' + ansi_bold + header + ansi_normal )
# end

#-------------------------------------------------------------------------------
def print_subhead( subhead ):
    '''
    Prints a subhead, both to console and the log.
    '''
    print( '-'*40 + '\n' +
           subhead, file=log )
    print( subhead )
# end

#-------------------------------------------------------------------------------
def print_msg( msg ):
    '''
    Prints msg, both to console and the log.
    '''
    print( msg, file=log )
    print( msg )
# end

#-------------------------------------------------------------------------------
def print_warn( msg ):
    '''
    Prints warning msg, with bold red font, both to console and the log.
    '''
    print( ansi_bold + ansi_red + 'Warning: ' + msg + ansi_normal, file=log )
    print( ansi_bold + ansi_red + 'Warning: ' + msg + ansi_normal )
# end

#-------------------------------------------------------------------------------
def print_test( label ):
    '''
    If label is given, prints the label, both to console and the log.
    On the console, it doesn't print the trailing newline; a subsequent
    print_result() will print it.
    If no label is given, does nothing. This simplifies functions like
    compile_obj that take an optional label to print.
    '''
    if (label):
        print( '-'*20 + '\n' + label, file=log )
        print( '%-72s' % label, end='' )
        sys.stdout.flush()
# end

#-------------------------------------------------------------------------------
def print_result( label, rc, extra='' ):
    '''
    If label is given, prints either "yes" (if rc == 0) or "no" (otherwise).
    Extra is printed after yes or no.
    If no label is given, does nothing.
    @see print_test().
    '''
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
class Environments( object ):
    '''
    Manages stack of environments, which are dictionaries of name=value pairs.
    '''

    # ----------------------------------------
    def __init__( self ):
        '''
        Initializes the environment stack.
        The bottom is os.environ. The top is an empty environment.
        '''
        self.stack = [ os.environ, {} ]

    # ----------------------------------------
    def push( self, env=None ):
        '''
        Push an empty enviroment on the environment stack.
        If env is given, also merge env into the environment stack.
        '''
        self.stack.append( {} )
        if (env):
            self.merge( env )

    # ----------------------------------------
    def top( self ):
        '''
        Return top-most environment in the environment stack.
        '''
        return self.stack[-1]

    # ----------------------------------------
    def pop( self ):
        '''
        Remove the top-most environment from the environment stack.
        '''
        if (len(self.stack) == 2):
            raise Error( "can't pop last 2 environments" )
        return self.stack.pop()

    # ----------------------------------------
    def __contains__( self, key ):
        '''
        Returns true if a key exists in the environment stack.
        '''
        for env in self.stack[::-1]:
            if (key in env):
                return True
        return False

    # ----------------------------------------
    def __getitem__( self, key ):
        '''
        Returns the value of the key, searching from the top of the environment
        stack down. As in a Makefile, unknown keys return empty string ('').
        Use 'x in environ' to test whether a key exists.
        '''
        for env in self.stack[::-1]:
            if (key in env):
                return env[ key ]
        return ''

    # ----------------------------------------
    def __setitem__( self, key, value ):
        '''
        Sets the key's value
        in the top-most environment in the environment stack.
        '''
        self.stack[ -1 ][ key ] = value

    # ----------------------------------------
    def append( self, key, val ):
        '''
        Append val to key's value, saving the result
        in the top-most environment in the enviornment stack.
        '''
        orig = self[ key ]
        if (val):
            if (orig):
                val = orig + ' ' + val
            self[ key ] = val
        return orig

    # ----------------------------------------
    def prepend( self, key, val ):
        '''
        Prepend val to key's value, saving the result
        in the top-most environment in the enviornment stack.
        '''
        orig = self[ key ]
        if (val):
            if (orig):
                val = val + ' ' + orig
            self[ key ] = val
        return orig

    # ----------------------------------------
    def merge( self, env ):
        '''
        Merges env, a dictionary of environment variables, into the existing
        environment stack. For most variables, the value in env is appended
        to any existing value. For LIBS, the value is prepended.
        For variables in config.replace_vars (like CXX), the value in env
        replaces the existing value.
        '''
        for key in env:
            if (key in replace_vars):
                self[ key ] = env[ key ]
            elif (key == 'LIBS'):
                self.prepend( key, env[ key ] )
            else:
                self.append( key, env[ key ] )
# end

#-------------------------------------------------------------------------------
def choose( prompt, choices ):
    '''
    Asks the user to choose among the given choices.
    Returns the index of the chosen item in the range [0, len(choices)-1],
    or raises Error or Quit exceptions.
    '''
    choices = list( choices )
    n = len( choices )
    if (n == 0):
        print( ansi_bold + ansi_red + 'none found' + ansi_normal )
        raise Error
    elif (n == 1):
        ##print()
        return 0
    else:
        width = int( math.log10( n ) + 1 )
        print( '\n' + prompt )
        for i in range( n ):
            print( '[%*d] %s' % (width, i+1, choices[i]) )
        while (True):
            print( 'Enter [1-%d] or quit: ' % (n), end='' )
            sys.stdout.flush()
            i = input()
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
    '''
    Runs the command cmd.
    cmd can be a string or a nested list.
    Pushes env beforehand and pops afterward.
    stdout and stderr are written to the log.
    Returns (return_code, stdout, stderr) from the command.

    Ex: run( ['gcc', '-c', 'file.c'], {'CPATH': '/opt/include'} )
    runs: gcc -c file.c
    '''
    environ.push( env )

    if (not isinstance( cmd, str )):
        cmd = ' '.join( flatten( cmd ))

    print( '>>>', cmd, file=log )
    cmd_list = shlex.split( cmd )
    try:
        proc = subprocess.Popen( cmd_list, stdout=PIPE, stderr=PIPE )
        (stdout, stderr) = proc.communicate()
        stdout = stdout.decode('utf-8')
        stderr = stderr.decode('utf-8')

        rc = proc.wait()
        log.write( stdout )
        if (stderr):
            log.write( ansi_red )
            log.write( stderr )
            log.write( ansi_normal )
        print( 'exit status = %d' % rc, file=log )
    except Exception as ex:
        print( 'Exception:', str(ex), file=log )
        rc = -1
        stdout = ''
        stderr = str(ex)

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
def compile_obj( src, env=None, label=None ):
    '''
    Compiles source file src into an object (.o) file.
    Pushes env beforehand and pops afterwards.
    If label is given, prints label & result.
    Returns (return_code, stdout, stderr) from the compiler.

    Ex: compile_obj( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
    runs: gcc $CFLAGS -c foo.c -o foo.o
    '''
    environ.push( env )

    print_test( label )
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
def link_exe( src, env=None, label=None ):
    '''
    Links the object file (.o) associated with the source file src into an executable.
    Assumes compile_obj( src ) was called previously to generate the object file.
    Pushes env beforehand and pops afterward.
    If label is given, prints label & result.
    Returns (return_code, stdout, stderr) from the compiler.

    Ex: link_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
    runs: gcc $LDFLAGS $LIBS foo.o -o foo
    '''
    environ.push( env )

    print_test( label )
    (base, ext) = os.path.splitext( src )
    obj      = base + '.o'
    lang     = lang_map[ ext ]
    compiler = environ[ lang ]
    LDFLAGS  = environ['LDFLAGS']
    LIBS     = environ['LIBS'] or environ['LDLIBS']
    (rc, stdout, stderr) = run([ compiler, obj, '-o', base, LDFLAGS, LIBS ])
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
def compile_exe( src, env=None, label=None ):
    '''
    Compiles source file src into an object file via compile_obj(),
    then links it into an exe.

    Ex: compile_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
    runs: gcc $CFLAGS -c foo.c -o foo.o
          gcc $LDFLAGS $LIBS foo.o -o foo
    '''
    environ.push( env )

    print_test( label )
    (base, ext) = os.path.splitext( src )
    obj      = base + '.o'
    lang     = lang_map[ ext ]
    compiler = environ[ lang ]
    LDFLAGS  = environ['LDFLAGS']
    LIBS     = environ['LIBS'] or environ['LDLIBS']
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
    '''
    Compiles source file src into an object file and exe via compile_exe(),
    then executes the exe.

    Ex: compile_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
    runs: gcc $CFLAGS -c foo.c -o foo.o
          gcc $LDFLAGS $LIBS foo.o -o foo
          ./foo
    '''
    environ.push( env )

    print_test( label )
    (base, ext) = os.path.splitext( src )
    (rc, stdout, stderr) = compile_exe( src )
    if (rc == 0):
        (rc, stdout, stderr) = run( './' + base )
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
def run_exe( src, env=None, label=None ):
    '''
    Runs the exe associated with src.
    Assumes compile_exe( src ) was called previously to generate the exe.

    Ex: run_exe( 'foo.c', {'CC': 'gcc'}, 'Test foo' )
    runs: ./foo
    '''
    environ.push( env )

    print_test( label )
    (base, ext) = os.path.splitext( src )
    (rc, stdout, stderr) = run( './' + base )
    print_result( label, rc )

    environ.pop()
    return (rc, stdout, stderr)
# end

#-------------------------------------------------------------------------------
def prog_cxx( choices=['g++', 'c++', 'CC', 'cxx', 'icpc', 'xlc++', 'clang++'] ):
    '''
    Searches for available C++ compilers from the list of choices.
    Sets CXX to the chosen one.
    '''
    print_header( 'C++ compiler' )

    cxx = environ['CXX']
    if (cxx):
        print( 'Trying $CXX =', cxx )
        choices = [ cxx ]

    passed = []
    for cxx in choices:
        print_test( cxx )
        (rc, out, err) = compile_run( 'config/compiler_cxx.cc', {'CXX': cxx} )
        # print (g++), (clang++), etc., as output by compiler_cxx, after yes
        if (rc == 0):
            out = '(' + out.strip() + ')'
        print_result( cxx, rc, out )
        if (rc == 0):
            passed.append( cxx )
            if (not interactive()): break
        # end
    # end
    i = choose( 'Choose C++ compiler:', passed )
    environ['CXX'] = passed[i]
# end

#-------------------------------------------------------------------------------
def prog_cxx_flags( flags ):
    '''
    Tests each flag in flags; if it passes, adds the flag to CXXFLAGS.
    '''
    print_header( 'C++ compiler flags' )
    for flag in flags:
        print_test( flag )
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
    If a flag works, it is added to both CXXFLAGS and LDFLAGS.
    '''
    print_header( 'OpenMP support' )
    src = 'config/openmp.cc'
    for flag in flags:
        print_test( flag )
        env = {'CXXFLAGS': flag, 'LDFLAGS': flag}
        (rc, out, err) = compile_run( src, env )
        print_result( flag, rc )
        if (rc == 0):
            environ.merge( env )
            break
    # end
# end

#-------------------------------------------------------------------------------
def get_package( name, directories, repo_url, tar_url, tar_filename ):
    '''
    Searches for a package, generally used for internal packages.
    Looks for a directory in directories; if found return directory.
    If not found, tries to 'hg clone repo_url' to the last directory.
    If that fails, tries to download tar_url and unpack it to the last directory.
    '''
    global log

    print_header( name )

    for directory in directories:
        print_test( directory )
        err = not os.path.exists( directory )
        print_result( directory, err )
        if (not err):
            return directory
    # end

    if (repo_url):
        if (interactive()):
            print( name +' not found; hg clone '+ repo_url +'? [Y/n] ', end='' )
            sys.stdout.flush()
            i = input().lower()
        if (not interactive() or i in ('', 'y', 'yes')):
            cmd = 'hg clone '+ repo_url +' '+ directory
            print_test( 'download: ' + cmd )
            (err, stdout, stderr) = run( cmd )
            print_result( 'download', err )
            if (not err):
                return directory
    # end

    if (tar_url):
        if (interactive()):
            print( name +' not found; download from '+ tar_url +'? [Y/n] ', end='' )
            sys.stdout.flush()
            i = input().lower()
        if (not interactive() or i in ('', 'y', 'yes')):
            try:
                print_test( 'download: '+ tar_url +' as '+ tar_filename )
                urlretrieve( tar_url, tar_filename )

                print( 'untar', tar_filename, file=log )
                tar = tarfile.open( tar_filename )
                files = tar.getnames()
                last = ''
                for f in files:
                    # sanitize file names: disallow beginning with / or having ../
                    if (re.search( r'^/|\.\./', f )):
                        print( 'skipping', f )
                        continue
                    tar.extract( f )
                    lastfile = f
                # end

                # rename directory,
                # e.g., from icl-testsweeper-dbd960ebf706 to testsweeper
                # todo: os.path.sep intsead of '/'?
                dirs = re.split( '/', lastfile )
                print( 'rename', dirs[0], directory, file=log )
                os.rename( dirs[0], directory )
                err = 0
            except Exception as ex:
                print( 'Exception:', str(ex), file=log )
            # end
            print_result( 'download', err )
            if (not err):
                return directory
        # end
    # end

    # otherwise, not found
    return None
# end

#-------------------------------------------------------------------------------
def extract_defines_from_flags( flags='CXXFLAGS' ):
    '''
    Extracts all "-Dname[=value]" defines from the given flags.
    Adds all "-Dname[=value]" defines to DEFINES.
    Adds all "#define name [value]" defines to HEADER_DEFINES.
    Stores all name=value defines for autoconf-like "#undef name" substitution
    in output_files().
    '''
    global environ, defines
    exp = r'(-D(\w+)(?:=(\S*))?) *'
    defs = re.findall( exp, environ[ flags ] )
    environ[ flags ] = re.sub( exp, '', environ[ flags ] ).strip()
    header = ''
    for (name_value, name, value) in defs:
        environ.append( 'DEFINES', name_value )
        defines[ name ] = value
        if (value):
            header += '#define '+ name +' '+ value + '\n'
        else:
            header += '#define '+ name + '\n'
    # end
    environ['HEADER_DEFINES'] = header
# end

#-------------------------------------------------------------------------------
def sub_env( match ):
    '''
    Given a re (regular expression) match object, returns value of environment variable.
    Used in output_files().
    '''
    return environ[ match.group(1) ]

#-------------------------------------------------------------------------------
def sub_define( match ):
    '''
    Given a re regexp match object,
    returns "#define name [value]" or "// #undef name"
    Used in output_files().
    '''
    global defines
    name = match.group(1)
    if (name in defines):
        value = defines[ name ]
        if (value):
            return '#define '+ name +' '+ value
        else:
            return '#define '+ name
    else:
        return '// #undef '+ name
# end

#-------------------------------------------------------------------------------
def read( filename ):
    '''
    Reads and returns the entire contents of filename.
    '''
    f = open( filename, 'r' )
    txt = f.read()
    f.close()
    return txt
# end

#-------------------------------------------------------------------------------
def write( filename, txt ):
    '''
    Writes txt to filename.
    '''
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
        txt = re.sub( r'@(\w+)@', sub_env, txt )
        txt = re.sub( r'#undef (\w+)', sub_define, txt )
        exists = os.path.exists( fname )
        if (exists and txt == read( fname )):
            print( fname, 'is unchanged' )
        else:
            if (exists):
                bak = fname + '.bak'
                print( 'backing up', fname, 'to', bak )
                os.rename( fname, bak )
            # end
            print( 'creating', fname )
            write( fname, txt )
        # end
    # end
# end

#-------------------------------------------------------------------------------
def init( prefix='/usr/local' ):
    '''
    Initializes config.
    Opens the logfile, deals with OS-specific issues, and parses command line
    options.
    '''
    global environ, log

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
            interactive( True )
        else:
            s = re.search( '^(\w+)=(.*)', arg )
            if (s):
                environ[ s.group(1) ] = s.group(2)
            else:
                print( 'Unknown argument:', arg )
                exit(1)
    # end

    if (environ['interactive'] == '1'):
        interactive( True )
# end

# ------------------------------------------------------------------------------
# Initialize global variables here, rather than in init(),
# so they are exported to __init__.py.
environ = Environments()
environ['argv'] = ' '.join( sys.argv )
environ['datetime'] = time.ctime()

defines = {}
