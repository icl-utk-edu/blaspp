#include "rotm.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_rotm_work( Params& params, bool run )
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
void test_rotm( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_rotm_work< int >( params, run );  // todo: generic implementation
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_rotm_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_rotm_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            //test_rotm_work< std::complex<float> >( params, run );  // not available for complex
            throw std::exception();
            break;

        case libtest::DataType::DoubleComplex:
            //test_rotm_work< std::complex<double> >( params, run );
            throw std::exception();
            break;
    }
}
