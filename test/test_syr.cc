#include "syr.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_syr_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T alpha = 123;

    blas::syr( blas::Layout::ColMajor, blas::Uplo::Lower, n,
               alpha, x, incx, A, lda );

    delete[] A;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_syr( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_syr_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_syr_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_syr_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_syr_work< std::complex<double> >( params, run );
            break;
    }
}
