#ifndef LAPACK_TMP_HH
#define LAPACK_TMP_HH

// get BLAS_FORTRAN_NAME and blas_int
#include "blas_fortran.hh"

#include <complex>

// This is a temporary file giving simple LAPACK wrappers,
// until the real lapackpp wrappers are available.

// -----------------------------------------------------------------------------
template < typename TX > inline
void lapack_larnv( blas_int idist, int iseed[4], blas_int size, TX *x )
{
    for (blas_int i = 0; i < size; ++i) {
        x[i] = rand() / TX ( RAND_MAX );
    }
}

template < typename TX > inline
void lapack_larnv( blas_int idist, int iseed[4], blas_int size, std::complex <TX> *x )
{
    for (blas_int i = 0; i < size; ++i) {
        x[i] = std::complex <TX>( rand() / TX ( RAND_MAX ), rand() / TX ( RAND_MAX ));
    }
}

// -----------------------------------------------------------------------------
float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work );

double lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work );

float  lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work );

double lapack_lange( char const *norm,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work );

double lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work );

float  lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work );

double lapack_lansy( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     float const *A, blas_int lda,
                     float *work );

double lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     double const *A, blas_int lda,
                     double *work );

float  lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work );

double lapack_lanhe( char const *norm, char const *uplo,
                     blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     float const *A, blas_int lda,
                     float *work );

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     double const *A, blas_int lda,
                     double *work );

float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<float> const *A, blas_int lda,
                     float *work );

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     blas_int m, blas_int n,
                     std::complex<double> const *A, blas_int lda,
                     double *work );

// -----------------------------------------------------------------------------
template< typename TA, typename TB > inline
void lapack_lacpy( char const* uplo,
                     blas_int m, blas_int n,
                     TA const *A, blas_int lda,
                     TB       *B, blas_int ldb )
{
    assert( tolower(*uplo) == 'g' );
    for (blas_int j = 0; j < n; ++j) {
        for (blas_int i = 0; i < m; ++i) {
            B[i + j * ldb] = A[i + j * lda];
        }
    }
}

// -----------------------------------------------------------------------------
void lapack_potrf(  char const *uplo, blas_int n,
                    float *A, blas_int lda,
                    blas_int *info );

void lapack_potrf(  char const *uplo, blas_int n,
                    double *A, blas_int lda,
                    blas_int *info );

void lapack_potrf(  char const *uplo, blas_int n,
                    std::complex<float> *A, blas_int lda,
                    blas_int *info );

void lapack_potrf(  char const *uplo, blas_int n,
                    std::complex<double> *A, blas_int lda,
                    blas_int *info );

#endif        //  #ifndef LAPACK_TMP_HH
