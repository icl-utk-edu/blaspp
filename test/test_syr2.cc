#include "syr2.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_syr2_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;

    blas::syr2( blas::Layout::ColMajor, blas::Uplo::Lower, n,
                alpha, x, incx, y, incy, A, lda );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_syr2( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_syr2_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_syr2_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_syr2_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_syr2_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_syr2_work< std::complex<double> >( params, run );
            break;
    }
}
