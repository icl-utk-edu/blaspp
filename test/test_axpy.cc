#include "axpy.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_axpy_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;
    
    blas::axpy( n, alpha, x, incx, y, incy );
    
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_axpy()
{
    printf( "\n%s\n", __func__ );
    test_axpy_work< int >();
    test_axpy_work< float >();
    test_axpy_work< double >();
    test_axpy_work< std::complex<float> >();
    test_axpy_work< std::complex<double> >();
}
