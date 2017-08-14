#include "nrm2.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_nrm2_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t incx = 1;
    T *x = new T[n];
    typename blas::traits<T>::norm_t result;

    result = blas::nrm2( n, x, incx );

    delete[] x;
}

// -----------------------------------------------------------------------------
void test_nrm2( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_nrm2_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_nrm2_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_nrm2_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_nrm2_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_nrm2_work< std::complex<double> >( params, run );
            break;
    }
}
