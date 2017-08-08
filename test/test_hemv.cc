#include "hemv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_hemv_work()
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;
    T beta  = 456;

    blas::hemv( blas::Layout::ColMajor, blas::Uplo::Lower, n,
                alpha, A, lda, x, incx, beta, y, incy );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_hemv()
{
    printf( "\n%s\n", __func__ );
    test_hemv_work< int >();
    test_hemv_work< float >();
    test_hemv_work< double >();
    test_hemv_work< std::complex<float> >();
    test_hemv_work< std::complex<double> >();
}
