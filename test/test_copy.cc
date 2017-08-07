#include "copy.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_copy_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    
    blas::copy( n, x, incx, y, incy );
    
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_copy()
{
    printf( "\n%s\n", __func__ );
    test_copy_work< int >();
    test_copy_work< float >();
    test_copy_work< double >();
    test_copy_work< std::complex<float> >();
    test_copy_work< std::complex<double> >();
}
