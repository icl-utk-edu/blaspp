#include "trmv.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_trmv_work()
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
void test_trmv()
{
    printf( "\n%s\n", __func__ );
    test_trmv_work< int >();
    test_trmv_work< float >();
    test_trmv_work< double >();
    test_trmv_work< std::complex<float> >();
    test_trmv_work< std::complex<double> >();
}
