#include <stdio.h>

#if defined(HAVE_ESSL)
    #include <essl.h>
#elif defined(HAVE_MKL)
    #include <mkl_cblas.h>
#else
    #include <cblas.h>
#endif

int main()
{
    int n = 5;
    double x[] = { 1, 2, 3, 4, 5 };
    double y[] = { 5, 4, 3, 2, 1 };
    double result = cblas_ddot( n, x, 1, y, 1 );
    bool okay = (result == 35);
    printf( "%s\n", okay ? "ok" : "failed" );
    return ! okay;
}
