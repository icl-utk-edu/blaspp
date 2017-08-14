#include "iamax.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_iamax_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    int64_t result;

    result = blas::iamax( n, x, incx );

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_iamax( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_iamax_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_iamax_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_iamax_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_iamax_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_iamax_work< std::complex<double> >( params, run );
            break;
    }
}
