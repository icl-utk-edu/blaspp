#ifndef PRINT_HH
#define PRINT_HH

#include <assert.h>
#include <complex>

// -----------------------------------------------------------------------------
template< typename T >
void print_matrix( int64_t m, int64_t n, T *A, int64_t lda )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf( " %9.4f", A(i,j) );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
template< typename T >
void print_matrix( int64_t m, int64_t n, std::complex<T>* A, int64_t lda )
{
    #define A(i_, j_) A[ (i_) + size_t(lda)*(j_) ]

    assert( m >= 0 );
    assert( n >= 0 );
    assert( lda >= m );

    printf( "[\n" );
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            printf( " %9.4f + %9.4fi", real(A(i,j)), imag(A(i,j)) );
        }
        printf( "\n" );
    }
    printf( "];\n" );

    #undef A
}

// -----------------------------------------------------------------------------
template< typename T >
void print_vector( int64_t n, T *x, int64_t incx )
{
    assert( n >= 0 );
    assert( incx != 0 );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        printf( " %9.4f", x[ix] );
        ix += incx;
    }
    printf( " ]\n" );
}

// -----------------------------------------------------------------------------
template< typename T >
void print_vector( int64_t n, std::complex<T>* x, int64_t incx )
{
    assert( n >= 0 );
    assert( incx != 0 );

    printf( "[" );
    int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
    for (int64_t i = 0; i < n; ++i) {
        printf( " %9.4f + %9.4fi", real(x[ix]), imag(x[ix]) );
        ix += incx;
    }
    printf( " ];\n" );
}

#endif        //  #ifndef PRINT_HH
