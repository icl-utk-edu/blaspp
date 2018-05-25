#include <stdio.h>
#include <mkl.h>

int main()
{
    MKLVersion v;
    MKL_Get_Version( &v );
    printf( "MKL_VERSION=%d.%d.%d\n",
            v.MajorVersion, v.MinorVersion, v.UpdateVersion );
    return 0;
}
