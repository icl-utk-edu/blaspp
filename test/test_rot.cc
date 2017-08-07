#include "rot.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rot_work()
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;
    typename blas::traits<T>::norm_t c = 1;
    T s = 0;
    
    blas::rot( n, x, incx, y, incy, c, s );
    
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_rot()
{
    printf( "\n%s\n", __func__ );
    //test_rot_work< int >();  // todo: generic implementation
    test_rot_work< float >();
    test_rot_work< double >();
    test_rot_work< std::complex<float> >();
    test_rot_work< std::complex<double> >();
}
