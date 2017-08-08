#include "her.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_her_work()
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    typename blas::traits<T>::norm_t alpha = 123;

    blas::her( blas::Layout::ColMajor, blas::Uplo::Lower, n,
               alpha, x, incx, A, lda );

    delete[] A;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_her()
{
    printf( "\n%s\n", __func__ );
    test_her_work< int >();
    test_her_work< float >();
    test_her_work< double >();
    test_her_work< std::complex<float> >();
    test_her_work< std::complex<double> >();
}
