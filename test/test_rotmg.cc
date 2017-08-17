#include "rotmg.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotmg_work( Params& params, bool run )
{
    T d1=1, d2=2, x1=3, y1=4;
    T ps[5] = { 0, 0, 0, 0, 0 };

    blas::rotmg( &d1, &d2, &x1, y1, ps );
}

// -----------------------------------------------------------------------------
void test_rotmg( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_rotmg_work< int64_t >( params, run );  // todo: generic implementation
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_rotmg_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_rotmg_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            //test_rotmg_work< std::complex<float> >( params, run );  // not available for complex
            throw std::exception();
            break;

        case libtest::DataType::DoubleComplex:
            //test_rotmg_work< std::complex<double> >( params, run );
            throw std::exception();
            break;
    }
}
