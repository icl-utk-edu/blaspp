#include "trsv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_trsv_work()
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
void test_trsv()
{
    printf( "\n%s\n", __func__ );
    test_trsv_work< int >();
    test_trsv_work< float >();
    test_trsv_work< double >();
    test_trsv_work< std::complex<float> >();
    test_trsv_work< std::complex<double> >();
}
