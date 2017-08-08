#include "syr.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_syr_work()
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T alpha = 123;

    blas::syr( blas::Layout::ColMajor, blas::Uplo::Lower, n,
               alpha, x, incx, A, lda );

    delete[] A;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_syr()
{
    printf( "\n%s\n", __func__ );
    test_syr_work< int >();
    test_syr_work< float >();
    test_syr_work< double >();
    test_syr_work< std::complex<float> >();
    test_syr_work< std::complex<double> >();
}
