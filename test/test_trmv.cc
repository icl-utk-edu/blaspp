#include "trmv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_trmv_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    T *A = new T[lda*n];
    T *x = new T[n];

    blas::trmv( blas::Layout::ColMajor, blas::Uplo::Lower,
                blas::Op::NoTrans, blas::Diag::NonUnit,
                n, A, lda, x, incx );

    delete[] A;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_trmv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_trmv_work< int >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_trmv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_trmv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_trmv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_trmv_work< std::complex<double> >( params, run );
            break;
    }
}
