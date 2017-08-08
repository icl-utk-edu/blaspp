#include "ger.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_ger_work()
{
    int64_t m = 200;
    int64_t lda = m;
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[m];
    T *y = new T[n];
    T alpha = 123;

    blas::ger( blas::Layout::ColMajor, m, n,
               alpha, x, incx, y, incy, A, lda );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_ger()
{
    printf( "\n%s\n", __func__ );
    test_ger_work< int >();
    test_ger_work< float >();
    test_ger_work< double >();
    test_ger_work< std::complex<float> >();
    test_ger_work< std::complex<double> >();
}
