#include "gemv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_gemv_work()
{
    int64_t m = 200;
    int64_t lda = m;
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T *y = new T[m];
    T alpha = 123;
    T beta  = 456;

    blas::gemv( blas::Layout::ColMajor, blas::Op::NoTrans, m, n,
                alpha, A, lda, x, incx, beta, y, incy );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_gemv()
{
    printf( "\n%s\n", __func__ );
    test_gemv_work< int >();
    test_gemv_work< float >();
    test_gemv_work< double >();
    test_gemv_work< std::complex<float> >();
    test_gemv_work< std::complex<double> >();
}
