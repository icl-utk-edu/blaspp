#include "syr2.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_syr2_work()
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;

    blas::syr2( blas::Layout::ColMajor, blas::Uplo::Lower, n,
                alpha, x, incx, y, incy, A, lda );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_syr2()
{
    printf( "\n%s\n", __func__ );
    test_syr2_work< int >();
    test_syr2_work< float >();
    test_syr2_work< double >();
    test_syr2_work< std::complex<float> >();
    test_syr2_work< std::complex<double> >();
}