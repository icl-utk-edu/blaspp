#include "rotg.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotg_work( Params& params, bool run )
{
    T a=1, b=2, s;
    typename blas::traits<T>::real_t c;

    blas::rotg( &a, &b, &c, &s );
}

// -----------------------------------------------------------------------------
void test_rotg( Params& params, bool run )
{
    switch (params.datatype.value()) {
       case libtest::DataType::Integer:
           //test_rotg_work< int64_t >( params, run );  // todo: generic implementation
           throw std::exception();
           break;

        case libtest::DataType::Single:
            test_rotg_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_rotg_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_rotg_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_rotg_work< std::complex<double> >( params, run );
            break;
    }
}
