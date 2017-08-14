#include "asum.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_asum_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    T result;

    result = blas::asum( n, x, incx );

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_asum( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_asum_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_asum_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_asum_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_asum_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_asum_work< std::complex<double> >( params, run );
            break;
    }
}
