#include "gemm.hh"
#include "test.hh"

// -----------------------------------------------------------------------------
template< typename T >
void test_gemm_work( Params& params, bool run )
{
    int64_t m = 200;
    int64_t n = 100;
    int64_t k = 100;
    blas::Op transA = params.transA.value();
    blas::Op transB = params.transB.value();
    int64_t Am, An, Bm, Bn;
    if (transA == blas::Op::NoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transB == blas::Op::NoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }
    int64_t lda = Am;
    int64_t ldb = Bm;
    int64_t ldc = m;

    T *A = new T[lda*An];
    T *B = new T[ldb*Bn];
    T *C = new T[ldc*n];
    T alpha = 123;
    T beta  = 456;

    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, n, k, alpha, A, lda, B, ldb, beta, C, ldc );

    delete[] A;
    delete[] B;
    delete[] C;
}

// -----------------------------------------------------------------------------
void test_gemm( Params& params, bool run )
{
    switch (params.datatype.value()) {
        case libtest::DataType::Integer:
            test_gemm_work< int >( params, run );
            break;

        case libtest::DataType::Single:
            test_gemm_work< float >( params, run );
            break;

        case libtest::DataType::Double:
            test_gemm_work< double >( params, run );
            break;

        case libtest::DataType::SingleComplex:
            test_gemm_work< std::complex<float> >( params, run );
            break;

        case libtest::DataType::DoubleComplex:
            test_gemm_work< std::complex<double> >( params, run );
            break;
    }
}
