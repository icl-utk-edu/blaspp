#include "dot.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_dot_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    T result;
    
    result = blas::dot( n, x, incx, y, incy );
    result = blas::dotu( n, x, incx, y, incy );
    
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_dot()
{
    printf( "\n%s\n", __func__ );
    test_dot_work< int >();
    test_dot_work< float >();
    test_dot_work< double >();
    test_dot_work< std::complex<float> >();
    test_dot_work< std::complex<double> >();
}
