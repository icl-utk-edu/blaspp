#include <blas.hh>

#include <vector>
#include <stdio.h>

//------------------------------------------------------------------------------
template <typename T>
void run( int m, int n, int k )
{
    int lda = m;
    int ldb = n;
    int ldc = m;
    std::vector<T> A( lda*k, 1.0 );  // m-by-k
    std::vector<T> B( ldb*n, 2.0 );  // k-by-n
    std::vector<T> C( ldc*n, 3.0 );  // m-by-n

    // ... fill in application data into A, B, C ...

    // C = -1.0*A*B + 1.0*C
    blas::gemm( blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                m, n, k,
                -1.0, A.data(), lda,
                      B.data(), ldb,
                 1.0, C.data(), ldc );
}

//------------------------------------------------------------------------------
int main( int argc, char** argv )
{
    int m = 100, n = 200, k = 50;
    printf( "run< float >( %d, %d, %d )\n", m, n, k );
    run< float  >( m, n, k );

    printf( "run< double >( %d, %d, %d )\n", m, n, k );
    run< double >( m, n, k );

    printf( "run< complex<float> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<float>  >( m, n, k );

    printf( "run< complex<double> >( %d, %d, %d )\n", m, n, k );
    run< std::complex<double> >( m, n, k );

    return 0;
}
