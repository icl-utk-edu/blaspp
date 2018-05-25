#include <stdio.h>
#include <openblas_config.h>

int main()
{
    const char* v = OPENBLAS_VERSION;
    printf( "OPENBLAS_VERSION=%s\n", v );
    return 0;
}
