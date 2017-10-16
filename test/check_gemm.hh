#ifndef CHECK_GEMM_HH
#define CHECK_GEMM_HH

#include "blas_util.hh"
#include "lapack.hh"

// -----------------------------------------------------------------------------
// Computes error for multiplication with general matrix result.
// Covers dot, gemv, ger, geru, gemm, symv, hemv, symm, trmv, trsv?, trmm, trsm?.
// Cnorm is norm of original C, before multiplication operation.
template< typename T >
void check_gemm(
    int64_t m, int64_t n, int64_t k,
    T alpha,
    T beta,
    typename blas::traits<T>::real_t Anorm,
    typename blas::traits<T>::real_t Bnorm,
    typename blas::traits<T>::real_t Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    typename blas::traits<T>::real_t error[1],
    int64_t* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    typedef typename blas::traits<T>::real_t real_t;

    assert( m >= 0 );
    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= m );
    assert( ldcref >= m );

    // C -= Cref
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            C(i,j) -= Cref(i,j);
        }
    }

    real_t work[1];
    error[0] = lapack_lange( "f", m, n, C, ldc, work )
             / (sqrt(real_t(k)+2)*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    real_t eps = std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < 3*eps);

    #undef C
    #undef Cref
}

// -----------------------------------------------------------------------------
// Computes error for multiplication with symmetric or Hermitian matrix result.
// Covers syr, syr2, syrk, syr2k, her, her2, herk, her2k.
// Cnorm is norm of original C, before multiplication operation.
//
// alpha and beta are either real or complex, depending on routine:
//          zher    zher2   zherk   zher2k  zsyr    zsyr2   zsyrk   zsyr2k
// alpha    real    complex real    complex complex complex complex complex
// beta     --      --      real    real    --      --      complex complex
// zsyr2 doesn't exist in standard BLAS or LAPACK.
template< typename TA, typename TB, typename T >
void check_herk(
    blas::Uplo uplo,
    int64_t n, int64_t k,
    TA alpha,
    TB beta,
    typename blas::traits<T>::real_t Anorm,
    typename blas::traits<T>::real_t Bnorm,
    typename blas::traits<T>::real_t Cnorm,
    T const* Cref, int64_t ldcref,
    T* C, int64_t ldc,
    typename blas::traits<T>::real_t error[1],
    int64_t* okay )
{
    #define    C(i_, j_)    C[ (i_) + (j_)*ldc ]
    #define Cref(i_, j_) Cref[ (i_) + (j_)*ldcref ]

    typedef typename blas::traits<T>::real_t real_t;

    assert( n >= 0 );
    assert( k >= 0 );
    assert( ldc >= n );
    assert( ldcref >= n );

    // C -= Cref
    if (uplo == blas::Uplo::Lower) {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = j; i < n; ++i) {
                C(i,j) -= Cref(i,j);
            }
        }
    }
    else {
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= j; ++i) {
                C(i,j) -= Cref(i,j);
            }
        }
    }

    real_t work[1];
    error[0] = lapack_lanhe( "f", uplo2str(uplo), n, C, ldc, work )
             / (sqrt(real_t(k)+2)*fabs(alpha)*Anorm*Bnorm + 2*fabs(beta)*Cnorm);

    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    real_t eps = std::numeric_limits< real_t >::epsilon();
    *okay = (error[0] < 3*eps);

    #undef C
    #undef Cref
}

#endif        //  #ifndef CHECK_GEMM_HH
