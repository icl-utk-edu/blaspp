#ifndef LAPACK_TMP_HH
#define LAPACK_TMP_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas_fortran.hh"

#include <complex>

// This is a temporary file giving simple LAPACK wrappers,
// until the real lapackpp wrappers are available.

// -----------------------------------------------------------------------------
#define lapack_slarnv BLAS_FORTRAN_NAME( slarnv, SLARNV )
#define lapack_dlarnv BLAS_FORTRAN_NAME( dlarnv, DLARNV )
#define lapack_clarnv BLAS_FORTRAN_NAME( clarnv, CLARNV )
#define lapack_zlarnv BLAS_FORTRAN_NAME( zlarnv, ZLARNV )

extern "C"
void lapack_slarnv( blas_int const* idist, blas_int iseed[4],
                    blas_int const* size, float *x );

extern "C"
void lapack_dlarnv( blas_int const* idist, blas_int iseed[4],
                    blas_int const* size, double *x );

extern "C"
void lapack_clarnv( blas_int const* idist, blas_int iseed[4],
                    blas_int const* size, std::complex<float> *x );

extern "C"
void lapack_zlarnv( blas_int const* idist, blas_int iseed[4],
                    blas_int const* size, std::complex<double> *x );

// -----------------------------------------------------------------------------
inline
void lapack_larnv( blas_int idist, blas_int iseed[4], blas_int size, float *x )
{
    lapack_slarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( blas_int idist, blas_int iseed[4], blas_int size, double *x )
{
    lapack_dlarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( blas_int idist, blas_int iseed[4], blas_int size, std::complex<float> *x )
{
    lapack_clarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( blas_int idist, blas_int iseed[4], blas_int size, std::complex<double> *x )
{
    lapack_zlarnv( &idist, iseed, &size, x );
}

// -----------------------------------------------------------------------------
#define lapack_slange BLAS_FORTRAN_NAME( slange, SLANGE )
#define lapack_dlange BLAS_FORTRAN_NAME( dlange, DLANGE )
#define lapack_clange BLAS_FORTRAN_NAME( clange, CLANGE )
#define lapack_zlange BLAS_FORTRAN_NAME( zlange, ZLANGE )

extern "C"
blas_float_return
       lapack_slange( char const *norm,
                      blas_int const *m, blas_int const *n,
                      float const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_dlange( char const *norm,
                      blas_int const *m, blas_int const *n,
                      double const *A, blas_int const *lda,
                      double *work );

extern "C"
blas_float_return
       lapack_clange( char const *norm,
                      blas_int const *m, blas_int const *n,
                      std::complex<float> const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_zlange( char const *norm,
                      blas_int const *m, blas_int const *n,
                      std::complex<double> const *A, blas_int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slange( norm, &m, &n, A, &lda, work );
}

inline
double lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlange( norm, &m, &n, A, &lda, work );
}

inline
float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clange( norm, &m, &n, A, &lda, work );
}

inline
double lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work )
{
    return lapack_zlange( norm, &m, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slansy BLAS_FORTRAN_NAME( slansy, SLANSY )
#define lapack_dlansy BLAS_FORTRAN_NAME( dlansy, DLANSY )
#define lapack_clansy BLAS_FORTRAN_NAME( clansy, CLANSY )
#define lapack_zlansy BLAS_FORTRAN_NAME( zlansy, ZLANSY )

extern "C"
blas_float_return
       lapack_slansy( char const *norm, char const *uplo,
                      blas_int const *n,
                      float const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_dlansy( char const *norm, char const *uplo,
                      blas_int const *n,
                      double const *A, blas_int const *lda,
                      double *work );

extern "C"
blas_float_return
       lapack_clansy( char const *norm, char const *uplo,
                      blas_int const *n,
                      std::complex<float> const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_zlansy( char const *norm, char const *uplo,
                      blas_int const *n,
                      std::complex<double> const *A, blas_int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

inline
float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work )
{
    return lapack_zlansy( norm, uplo, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_clanhe BLAS_FORTRAN_NAME( clanhe, CLANHE )
#define lapack_zlanhe BLAS_FORTRAN_NAME( zlanhe, ZLANHE )

extern "C"
blas_float_return
       lapack_clanhe( char const *norm, char const *uplo,
                      blas_int const *n,
                      std::complex<float> const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_zlanhe( char const *norm, char const *uplo,
                      blas_int const *n,
                      std::complex<double> const *A, blas_int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

inline
float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clanhe( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work )
{
    return lapack_zlanhe( norm, uplo, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slantr BLAS_FORTRAN_NAME( slantr, SLANTR )
#define lapack_dlantr BLAS_FORTRAN_NAME( dlantr, DLANTR )
#define lapack_clantr BLAS_FORTRAN_NAME( clantr, CLANTR )
#define lapack_zlantr BLAS_FORTRAN_NAME( zlantr, ZLANTR )

extern "C"
blas_float_return
       lapack_slantr( char const *norm, char const *uplo, char const *diag,
                      blas_int const *m, blas_int const *n,
                      float const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_dlantr( char const *norm, char const *uplo, char const *diag,
                      blas_int const *m, blas_int const *n,
                      double const *A, blas_int const *lda,
                      double *work );

extern "C"
blas_float_return
       lapack_clantr( char const *norm, char const *uplo, char const *diag,
                      blas_int const *m, blas_int const *n,
                      std::complex<float> const *A, blas_int const *lda,
                      float *work );

extern "C"
double lapack_zlantr( char const *norm, char const *uplo, char const *diag,
                      blas_int const *m, blas_int const *n,
                      std::complex<double> const *A, blas_int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work )
{
    return lapack_zlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slacpy BLAS_FORTRAN_NAME( slacpy, SLACPY )
#define lapack_dlacpy BLAS_FORTRAN_NAME( dlacpy, DLACPY )
#define lapack_clacpy BLAS_FORTRAN_NAME( clacpy, CLACPY )
#define lapack_zlacpy BLAS_FORTRAN_NAME( zlacpy, ZLACPY )

extern "C"
void lapack_slacpy( char const* uplo,
                    blas_int const *m, blas_int const *n,
                    float const *A, blas_int const *lda,
                    float const *B, blas_int const *ldb );

extern "C"
void lapack_dlacpy( char const* uplo,
                    blas_int const *m, blas_int const *n,
                    double const *A, blas_int const *lda,
                    double const *B, blas_int const *ldb );

extern "C"
void lapack_clacpy( char const* uplo,
                    blas_int const *m, blas_int const *n,
                    std::complex<float> const *A, blas_int const *lda,
                    std::complex<float> const *B, blas_int const *ldb );

extern "C"
void lapack_zlacpy( char const* uplo,
                    blas_int const *m, blas_int const *n,
                    std::complex<double> const *A, blas_int const *lda,
                    std::complex<double> const *B, blas_int const *ldb );

// -----------------------------------------------------------------------------
inline
void lapack_lacpy( char const* uplo,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float const *B, blas_int ldb )
{
    lapack_slacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double const *B, blas_int ldb )
{
    lapack_dlacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     std::complex<float> const *B, blas_int ldb )
{
    lapack_clacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     std::complex<double> const *B, blas_int ldb )
{
    lapack_zlacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

// -----------------------------------------------------------------------------
#define lapack_sgetrf BLAS_FORTRAN_NAME( sgetrf, SGETRF )
#define lapack_dgetrf BLAS_FORTRAN_NAME( dgetrf, DGETRF )
#define lapack_cgetrf BLAS_FORTRAN_NAME( cgetrf, CGETRF )
#define lapack_zgetrf BLAS_FORTRAN_NAME( zgetrf, ZGETRF )

extern "C"
void lapack_sgetrf( blas_int const *m, blas_int const *n,
                    float *A, blas_int const *lda,
                    blas_int *ipiv, blas_int *info );

extern "C"
void lapack_dgetrf( blas_int const *m, blas_int const *n,
                    double *A, blas_int const *lda,
                    blas_int *ipiv, blas_int *info );

extern "C"
void lapack_cgetrf( blas_int const *m, blas_int const *n,
                    std::complex<float> *A, blas_int const *lda,
                    blas_int *ipiv, blas_int *info );

extern "C"
void lapack_zgetrf( blas_int const *m, blas_int const *n,
                    std::complex<double> *A, blas_int const *lda,
                    blas_int *ipiv, blas_int *info );

// -----------------------------------------------------------------------------
inline
void lapack_getrf(  blas_int m, blas_int n,
                    float *A, blas_int lda,
                    blas_int *ipiv, blas_int *info )
{
    lapack_sgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  blas_int m, blas_int n,
                    double *A, blas_int lda,
                    blas_int *ipiv, blas_int *info )
{
    lapack_dgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  blas_int m, blas_int n,
                    std::complex<float> *A, blas_int lda,
                    blas_int *ipiv, blas_int *info )
{
    lapack_cgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  blas_int m, blas_int n,
                    std::complex<double> *A, blas_int lda,
                    blas_int *ipiv, blas_int *info )
{
    lapack_zgetrf( &m, &n, A, &lda, ipiv, info );
}

#endif        //  #ifndef LAPACK_TMP_HH
