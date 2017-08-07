#include "swap.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_swap_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    
    blas::swap( n, x, incx, y, incy );
    
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_swap()
{
    printf( "\n%s\n", __func__ );
    test_swap_work< int >();
    test_swap_work< float >();
    test_swap_work< double >();
    test_swap_work< std::complex<float> >();
    test_swap_work< std::complex<double> >();
}
