#include "scal.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_scal_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    T alpha = 123;

    blas::scal( n, alpha, x, incx );

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_scal( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_scal_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_scal_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_scal_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_scal_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_scal_work< std::complex<double> >( params, run );
            break;
    }
}
