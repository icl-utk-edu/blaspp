#include "iamax.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_iamax_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    int64_t result;
    
    result = blas::iamax( n, x, incx );
    
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_iamax()
{
    printf( "\n%s\n", __func__ );
    test_iamax_work< int >();
    test_iamax_work< float >();
    test_iamax_work< double >();
    test_iamax_work< std::complex<float> >();
    test_iamax_work< std::complex<double> >();
}
