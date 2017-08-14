#include "trsv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_trsv_work( Params& params, bool run )
{
    int64_t n = 100;
    int64_t lda = n;
    int64_t incx = 1;
    T *A = new T[lda*n];
    T *x = new T[n];

    blas::trsv( blas::Layout::ColMajor, blas::Uplo::Lower,
                blas::Op::NoTrans, blas::Diag::Unit,
                n, A, lda, x, incx );

    delete[] A;
    delete[] x;
}

// -----------------------------------------------------------------------------
void test_trsv( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            //test_trsv_work< int >( params, run );
            throw std::exception();
            break;

        case libtest::DataType::Single:
            test_trsv_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_trsv_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_trsv_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_trsv_work< std::complex<double> >( params, run );
            break;
    }
}
