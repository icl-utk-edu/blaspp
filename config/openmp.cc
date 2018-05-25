#include <omp.h>
#include <stdio.h>

int main()
{
    int nthreads = 1;
    int tid = 0;
    #pragma omp parallel
    {
        nthreads = omp_get_max_threads();
        tid = omp_get_thread_num();
        printf( "tid %d, nthreads %d\n", tid, nthreads );
    }
    printf( "ok\n" );
    return 0;
}
