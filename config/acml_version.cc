#include <stdio.h>
#include <mkl.h>

int main()
{
    int major, minor, patch, build;
    acmlversion( &major, &minor, &patch, &build );
    printf( "ACML_VERSION=%d.%d.%d.%d\n",
            major, minor, patch, build );
    return 0;
}
