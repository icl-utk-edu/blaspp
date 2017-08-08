#include "nrm2.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_nrm2_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    typename blas::traits<T>::norm_t result;

    result = blas::nrm2( n, x, incx );

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_nrm2()
{
    printf( "\n%s\n", __func__ );
    test_nrm2_work< int >();
    test_nrm2_work< float >();
    test_nrm2_work< double >();
    test_nrm2_work< std::complex<float> >();
    test_nrm2_work< std::complex<double> >();
}
