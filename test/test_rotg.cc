#include "rotg.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotg_work()
{
    T a=1, b=2, s;
    typename blas::traits<T>::norm_t c;

    blas::rotg( &a, &b, &c, &s );
}

// -----------------------------------------------------------------------------
void test_rotg()
{
    printf( "\n%s\n", __func__ );
    //test_rotg_work< int >();  // todo: generic implementation
    test_rotg_work< float >();
    test_rotg_work< double >();
    test_rotg_work< std::complex<float> >();
    test_rotg_work< std::complex<double> >();
}
