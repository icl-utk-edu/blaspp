#include "scal.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_scal_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    T alpha = 123;
    
    blas::scal( n, alpha, x, incx );
    
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_scal()
{
    printf( "\n%s\n", __func__ );
    test_scal_work< int >();
    test_scal_work< float >();
    test_scal_work< double >();
    test_scal_work< std::complex<float> >();
    test_scal_work< std::complex<double> >();
}
