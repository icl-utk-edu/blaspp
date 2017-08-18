#ifndef CBLAS_HH
#define CBLAS_HH

#include <cblas.h>

#include <complex>

#include "blas_util.hh"


// =============================================================================
// constants

// -----------------------------------------------------------------------------
typedef enum CBLAS_ORDER CBLAS_LAYOUT;

inline CBLAS_LAYOUT cblas_layout_const( blas::Layout layout )
{
    switch (layout) {
        case blas::Layout::RowMajor:  return CblasRowMajor;
        case blas::Layout::ColMajor:  return CblasColMajor;
        default: assert( false );
    }
}

inline CBLAS_LAYOUT cblas_layout_const( char layout )
{
    switch (layout) {
        case 'r': case 'R': return CblasRowMajor;
        case 'c': case 'C': return CblasColMajor;
        default:
            printf( "%s( %c )\n", __func__, layout );
            assert( false );
    }
}

inline char lapack_layout_const( CBLAS_LAYOUT layout )
{
    switch (layout) {
        case CblasRowMajor: return 'r';
        case CblasColMajor: return 'c';
        default: assert( false );
    }
}


// -----------------------------------------------------------------------------
typedef enum CBLAS_DIAG CBLAS_DIAG;

inline CBLAS_DIAG cblas_diag_const( blas::Diag diag )
{
    switch (diag) {
        case blas::Diag::NonUnit:  return CblasNonUnit;
        case blas::Diag::Unit:     return CblasUnit;
        default: assert( false );
    }
}

inline CBLAS_DIAG cblas_diag_const( char diag )
{
    switch (diag) {
        case 'n': case 'N': return CblasNonUnit;
        case 'u': case 'U': return CblasUnit;
        default:
            printf( "%s( %c )\n", __func__, diag );
            assert( false );
    }
}

inline char lapack_diag_const( CBLAS_DIAG diag )
{
    switch (diag) {
        case CblasNonUnit: return 'n';
        case CblasUnit: return 'u';
        default: assert( false );
    }
}


// -----------------------------------------------------------------------------
typedef enum CBLAS_SIDE CBLAS_SIDE;

inline CBLAS_SIDE cblas_side_const( blas::Side side )
{
    switch (side) {
        case blas::Side::Left:  return CblasLeft;
        case blas::Side::Right: return CblasRight;
        default: assert( false );
    }
}

inline CBLAS_SIDE cblas_side_const( char side )
{
    switch (side) {
        case 'l': case 'L': return CblasLeft;
        case 'r': case 'R': return CblasRight;
        default:
            printf( "%s( %c )\n", __func__, side );
            assert( false );
    }
}

inline char lapack_side_const( CBLAS_SIDE side )
{
    switch (side) {
        case CblasLeft:  return 'l';
        case CblasRight: return 'r';
        default: assert( false );
    }
}


// -----------------------------------------------------------------------------
typedef enum CBLAS_TRANSPOSE CBLAS_TRANSPOSE;

inline CBLAS_TRANSPOSE cblas_trans_const( blas::Op trans )
{
    switch (trans) {
        case blas::Op::NoTrans:   return CblasNoTrans;
        case blas::Op::Trans:     return CblasTrans;
        case blas::Op::ConjTrans: return CblasConjTrans;
        default: assert( false );
    }
}

inline CBLAS_TRANSPOSE cblas_trans_const( char trans )
{
    switch (trans) {
        case 'n': case 'N': return CblasNoTrans;
        case 't': case 'T': return CblasTrans;
        case 'c': case 'C': return CblasConjTrans;
        default:
            printf( "%s( %c )\n", __func__, trans );
            assert( false );
    }
}

inline char lapack_trans_const( CBLAS_TRANSPOSE trans )
{
    switch (trans) {
        case CblasNoTrans:   return 'n';
        case CblasTrans:     return 't';
        case CblasConjTrans: return 'c';
        default: assert( false );
    }
}


// -----------------------------------------------------------------------------
typedef enum CBLAS_UPLO CBLAS_UPLO;

inline CBLAS_UPLO cblas_uplo_const( blas::Uplo uplo )
{
    switch (uplo) {
        case blas::Uplo::Lower: return CblasLower;
        case blas::Uplo::Upper: return CblasUpper;
        default: assert( false );
    }
}

inline CBLAS_UPLO cblas_uplo_const( char uplo )
{
    switch (uplo) {
        case 'l': case 'L': return CblasLower;
        case 'u': case 'U': return CblasUpper;
        default:
            printf( "%s( %c )\n", __func__, uplo );
            assert( false );
    }
}

inline char lapack_uplo_const( CBLAS_UPLO uplo )
{
    switch (uplo) {
        case CblasLower: return 'l';
        case CblasUpper: return 'u';
        default:
            printf( "%s( %c )\n", __func__, uplo );
            assert( false );
    }
}


// =============================================================================
// Level 1 BLAS

// -----------------------------------------------------------------------------
inline float
cblas_asum(
    int n, float const *x, int incx )
{
    return cblas_sasum( n, x, incx );
}

inline double
cblas_asum(
    int n, double const *x, int incx )
{
    return cblas_dasum( n, x, incx );
}

inline float
cblas_asum(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scasum( n, x, incx );
}

inline double
cblas_asum(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dzasum( n, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_axpy(
    int n, float alpha,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_saxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, double alpha,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_daxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_caxpy( n, &alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zaxpy( n, &alpha, x, incx, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_copy(
    int n,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_scopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_dcopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_ccopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zcopy( n, x, incx, y, incy );
}


// -----------------------------------------------------------------------------
inline float
cblas_dot(
    int n,
    float const *x, int incx,
    float const *y, int incy )
{
    return cblas_sdot( n, x, incx, y, incy );
}

inline double
cblas_dot(
    int n,
    double const *x, int incx,
    double const *y, int incy )
{
    return cblas_ddot( n, x, incx, y, incy );
}

inline std::complex<float>
cblas_dot(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotc_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dot(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotc_sub( n, x, incx, y, incy, &result );
    return result;
}


// -----------------------------------------------------------------------------
// real dotu is same as dot
inline float
cblas_dotu(
    int n,
    float const *x, int incx,
    float const *y, int incy )
{
    return cblas_sdot( n, x, incx, y, incy );
}

inline double
cblas_dotu(
    int n,
    double const *x, int incx,
    double const *y, int incy )
{
    return cblas_ddot( n, x, incx, y, incy );
}

inline std::complex<float>
cblas_dotu(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dotu(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotu_sub( n, x, incx, y, incy, &result );
    return result;
}


// -----------------------------------------------------------------------------
inline float
cblas_iamax(
    int n, float const *x, int incx )
{
    return cblas_isamax( n, x, incx );
}

inline double
cblas_iamax(
    int n, double const *x, int incx )
{
    return cblas_idamax( n, x, incx );
}

inline float
cblas_iamax(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_icamax( n, x, incx );
}

inline double
cblas_iamax(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_izamax( n, x, incx );
}


// -----------------------------------------------------------------------------
inline float
cblas_nrm2(
    int n, float const *x, int incx )
{
    return cblas_snrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, double const *x, int incx )
{
    return cblas_dnrm2( n, x, incx );
}

inline float
cblas_nrm2(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scnrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dznrm2( n, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_scal(
    int n, float alpha,
    float* x, int incx )
{
    cblas_sscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, double alpha,
    double* x, int incx )
{
    cblas_dscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<float> alpha,
    std::complex<float>* x, int incx )
{
    cblas_cscal( n, &alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<double> alpha,
    std::complex<double>* x, int incx )
{
    cblas_zscal( n, &alpha, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_swap(
    int n,
    float* x, int incx,
    float* y, int incy )
{
    cblas_sswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    double* x, int incx,
    double* y, int incy )
{
    cblas_dswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<float>* x, int incx,
    std::complex<float>* y, int incy )
{
    cblas_cswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<double>* x, int incx,
    std::complex<double>* y, int incy )
{
    cblas_zswap( n, x, incx, y, incy );
}


// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_sgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_cgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_chemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zhemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

#define FORTRAN_NAME( lower, UPPER )  lower

#define lapack_csymv FORTRAN_NAME( csymv, CSYMV )
#define lapack_zsymv FORTRAN_NAME( zsymv, ZSYMV )

extern "C"
void   lapack_csymv(  const char *uplo,
                      const int *n,
                      const std::complex<float> *alpha,
                      const std::complex<float> *A, const int *lda,
                      const std::complex<float> *x, const int *incx,
                      const std::complex<float> *beta,
                            std::complex<float> *y, const int *incy );

extern "C"
void   lapack_zsymv(  const char *uplo,
                      const int *n,
                      const std::complex<double> *alpha,
                      const std::complex<double> *A, const int *lda,
                      const std::complex<double> *x, const int *incx,
                      const std::complex<double> *beta,
                            std::complex<double> *y, const int *incy );

inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    lapack_csymv( &uplo_, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}

inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    lapack_zsymv( &uplo_, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    cblas_cher( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    cblas_zher( layout, uplo, n, alpha, x, incx, A, lda );
}


// -----------------------------------------------------------------------------
#define lapack_csyr FORTRAN_NAME( csyr, CSYR )
#define lapack_zsyr FORTRAN_NAME( zsyr, ZSYR )

extern "C"
void   lapack_csyr(   const char *uplo,
                      const int *n,
                      const std::complex<float> *alpha,
                      const std::complex<float> *x, const int *incx,
                            std::complex<float> *A, const int *lda );

extern "C"
void   lapack_zsyr(   const char *uplo,
                      const int *n,
                      const std::complex<double> *alpha,
                      const std::complex<double> *x, const int *incx,
                            std::complex<double> *A, const int *lda );

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    lapack_csyr( &uplo_, &n, &alpha, x, &incx, A, &lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    char uplo_ = lapack_uplo_const( uplo );
    if (layout == CblasRowMajor) {
        uplo_ = (uplo == CblasUpper ? 'l' : 'u');  // switch upper <=> lower
    }
    lapack_zsyr( &uplo_, &n, &alpha, x, &incx, A, &lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_ssyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dsyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_ssyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dsyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    fprintf( stderr, "csyr2 unavailable\n" );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    fprintf( stderr, "zsyr2 unavailable\n" );
}


// =============================================================================
// Level 3 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_sgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_cgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_chemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zhemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_csymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}


// -----------------------------------------------------------------------------
#define lapack_csyr2k FORTRAN_NAME( csyr2k, CSYR2K )
#define lapack_zsyr2k FORTRAN_NAME( zsyr2k, ZSYR2K )

extern "C"
void   lapack_csyr2k( const char *uplo, const char *trans,
                      const int *n, const int *k,
                      const std::complex<float> *alpha,
                      const std::complex<float> *A, const int *lda,
                      const std::complex<float> *B, const int *ldb,
                      const std::complex<float> *beta,
                            std::complex<float> *C, const int *ldc );

extern "C"
void   lapack_zsyr2k( const char *uplo, const char *trans,
                      const int *n, const int *k,
                      const std::complex<double> *alpha,
                      const std::complex<double> *A, const int *lda,
                      const std::complex<double> *B, const int *ldb,
                      const std::complex<double> *beta,
                            std::complex<double> *C, const int *ldc );

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    if (layout == CblasRowMajor) {
        // swap upper <=> lower
        uplo = (uplo == CblasLower ? CblasUpper : CblasLower);
    }
    char uplo_  = lapack_uplo_const( uplo  );
    char trans_ = lapack_trans_const( trans );
    lapack_csyr2k( &uplo_, &trans_, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    if (layout == CblasRowMajor) {
        // swap upper <=> lower
        uplo = (uplo == CblasLower ? CblasUpper : CblasLower);
    }
    char uplo_  = lapack_uplo_const( uplo  );
    char trans_ = lapack_trans_const( trans );
    lapack_zsyr2k( &uplo_, &trans_, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    float beta,  // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    double beta,  // note: real
    std::complex<double>* C, int ldc )
{
    cblas_cher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_rot(
    int n,
    float *x, int incx,
    float *y, int incy,
    float c, float s )
{
    cblas_srot( n, x, incx, y, incy, c, s );
}

inline void
cblas_rot(
    int n,
    double *x, int incx,
    double *y, int incy,
    double c, double s )
{
    cblas_drot( n, x, incx, y, incy, c, s );
}

inline void
cblas_rot(
    int n,
    std::complex<float> *x, int incx,
    std::complex<float> *y, int incy,
    float c, std::complex<float> s )
{
    throw std::exception();
    //cblas_crot( n, x, incx, y, incy, c, s );
}

inline void
cblas_rot(
    int n,
    std::complex<double> *x, int incx,
    std::complex<double> *y, int incy,
    double c, std::complex<double> s )
{
    throw std::exception();
    //cblas_zrot( n, x, incx, y, incy, c, s );
}

#endif        //  #ifndef CBLAS_HH
