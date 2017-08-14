#include "copy.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_copy_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    int64_t incy = 1;
    T *x = new T[n];
    T *y = new T[n];

    blas::copy( n, x, incx, y, incy );

    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_copy( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_copy_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_copy_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_copy_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_copy_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_copy_work< std::complex<double> >( params, run );
            break;
    }
}
