#ifndef UTIL_H
#define UTIL_H

//------------------------------------------------------------------------------
void print_func_( const char* func )
{
    printf( "\n%s\n", func );
}

#ifdef __GNUC__
    #define print_func() print_func_( __PRETTY_FUNCTION__ )
#else
    #define print_func() print_func_( __func__ )
#endif

#endif // UTIL_H
