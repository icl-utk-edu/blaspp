#include "rotmg.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotmg_work()
{
    T d1=1, d2=2, x1=3, y1=4;
    T params[5] = { 0, 0, 0, 0, 0 };
    
    blas::rotmg( &d1, &d2, &x1, y1, params );
}

// -----------------------------------------------------------------------------
void test_rotmg()
{
    printf( "\n%s\n", __func__ );
    //test_rotmg_work< int >();  // todo: generic implementation
    test_rotmg_work< float >();
    test_rotmg_work< double >();
    //test_rotmg_work< std::complex<float> >();  // not available for complex
    //test_rotmg_work< std::complex<double> >();
}
