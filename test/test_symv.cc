#include "symv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_symv_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    T *A = new T[lda*n];
    T *x = new T[n];
    T *y = new T[n];
    T alpha = 123;
    T beta  = 456;

    blas::symv( blas::Layout::ColMajor, blas::Uplo::Lower, n,
                alpha, A, lda, x, incx, beta, y, incy );

    delete[] A;
    delete[] x;
    delete[] y;
}

// -----------------------------------------------------------------------------
void test_symv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_symv_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_symv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_symv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_symv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_symv_work< std::complex<double> >( params, run );
            break;
    }
}
