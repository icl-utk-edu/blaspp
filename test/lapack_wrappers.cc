#ifndef LAPACK_TMP_HH
#define LAPACK_TMP_HH

// Moved from lapack_tmp.hh for ESSL compatability

// get BLAS_FORTRAN_NAME and blas_int

#include "blas_fortran.hh"
#include "lapack_tmp.hh"

#include <complex>

// This is a temporary file giving simple LAPACK wrappers,
// until the real lapackpp wrappers are available.

#define lapack_slange BLAS_FORTRAN_NAME( slange, SLANGE )
#define lapack_dlange BLAS_FORTRAN_NAME( dlange, DLANGE )
#define lapack_clange BLAS_FORTRAN_NAME( clange, CLANGE )
#define lapack_zlange BLAS_FORTRAN_NAME( zlange, ZLANGE )

// -----------------------------------------------------------------------------
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
float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slange( norm, &m, &n, A, &lda, work );
}

double lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlange( norm, &m, &n, A, &lda, work );
}

float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clange( norm, &m, &n, A, &lda, work );
}

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

// -----------------------------------------------------------------------------
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
float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

double lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clansy( norm, uplo, &n, A, &lda, work );
}

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

// -----------------------------------------------------------------------------
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
float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

double lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clanhe( norm, uplo, &n, A, &lda, work );
}

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

// -----------------------------------------------------------------------------
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
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work )
{
    return lapack_slantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work )
{
    return lapack_dlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work )
{
    return lapack_clantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work )
{
    return lapack_zlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_spotrf BLAS_FORTRAN_NAME( spotrf, SPOTRF )
#define lapack_dpotrf BLAS_FORTRAN_NAME( dpotrf, DPOTRF )
#define lapack_cpotrf BLAS_FORTRAN_NAME( cpotrf, CPOTRF )
#define lapack_zpotrf BLAS_FORTRAN_NAME( zpotrf, ZPOTRF )

// -----------------------------------------------------------------------------
extern "C"
void lapack_spotrf( char const *uplo, blas_int const *n,
                    float *A, blas_int const *lda,
                    blas_int *info );

extern "C"
void lapack_dpotrf( char const *uplo, blas_int const *n,
                    double *A, blas_int const *lda,
                    blas_int *info );

extern "C"
void lapack_cpotrf( char const *uplo, blas_int const *n,
                    std::complex<float> *A, blas_int const *lda,
                    blas_int *info );

extern "C"
void lapack_zpotrf( char const *uplo, blas_int const *n,
                    std::complex<double> *A, blas_int const *lda,
                    blas_int *info );

// -----------------------------------------------------------------------------
void lapack_potrf(  char const *uplo, blas_int n,
                    float *A, blas_int lda,
                    blas_int *info )
{
    lapack_spotrf( uplo, &n, A, &lda, info );
}

void lapack_potrf(  char const *uplo, blas_int n,
                    double *A, blas_int lda,
                    blas_int *info )
{
    lapack_dpotrf( uplo, &n, A, &lda, info );
}

void lapack_potrf(  char const *uplo, blas_int n,
                    std::complex<float> *A, blas_int lda,
                    blas_int *info )
{
    lapack_cpotrf( uplo, &n, A, &lda, info );
}

void lapack_potrf(  char const *uplo, blas_int n,
                    std::complex<double> *A, blas_int lda,
                    blas_int *info )
{
    lapack_zpotrf( uplo, &n, A, &lda, info );
}

#endif        //  #ifndef LAPACK_TMP_HH

