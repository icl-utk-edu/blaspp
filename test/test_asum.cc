#include "asum.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_asum_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    T result;
    
    result = blas::asum( n, x, incx );
    
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_asum()
{
    printf( "\n%s\n", __func__ );
    test_asum_work< int >();
    test_asum_work< float >();
    test_asum_work< double >();
    test_asum_work< std::complex<float> >();
    test_asum_work< std::complex<double> >();
}
