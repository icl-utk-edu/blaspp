#include "rotm.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotm_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    T param[5] = { 0, 0, 0, 0, 0 };

    blas::rotm( n, x, incx, y, incy, param );

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_rotm()
{
    printf( "\n%s\n", __func__ );
    //test_rotm_work< int >();  // todo: generic implementation
    test_rotm_work< float >();
    test_rotm_work< double >();
    //test_rotm_work< std::complex<float> >();  // not available for complex
    //test_rotm_work< std::complex<double> >();
}
