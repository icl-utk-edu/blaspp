#include "dot.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_dot_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];
    T result;

    result = blas::dot( n, x, incx, y, incy );
    result = blas::dotu( n, x, incx, y, incy );

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_dot( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_dot_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_dot_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_dot_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_dot_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_dot_work< std::complex<double> >( params, run );
            break;
    }
}
