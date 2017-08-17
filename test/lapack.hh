#ifndef LAPACK_HH
#define LAPACK_HH

// -----------------------------------------------------------------------------
// clapack / MacOS Accelerate return double instead of float
typedef double BLAS_RETURN_FLOAT;

// -----------------------------------------------------------------------------
#define lapack_slarnv FORTRAN_NAME( slarnv, SLARNV )
#define lapack_dlarnv FORTRAN_NAME( dlarnv, DLARNV )
#define lapack_clarnv FORTRAN_NAME( clarnv, CLARNV )
#define lapack_zlarnv FORTRAN_NAME( zlarnv, ZLARNV )

extern "C"
void lapack_slarnv( int const* idist, int iseed[4],
                    int const* size, float *x );

extern "C"
void lapack_dlarnv( int const* idist, int iseed[4],
                    int const* size, double *x );

extern "C"
void lapack_clarnv( int const* idist, int iseed[4],
                    int const* size, std::complex<float> *x );

extern "C"
void lapack_zlarnv( int const* idist, int iseed[4],
                    int const* size, std::complex<double> *x );

// -----------------------------------------------------------------------------
inline
void lapack_larnv( int idist, int iseed[4], int size, float *x )
{
    lapack_slarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( int idist, int iseed[4], int size, double *x )
{
    lapack_dlarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( int idist, int iseed[4], int size, std::complex<float> *x )
{
    lapack_clarnv( &idist, iseed, &size, x );
}

inline
void lapack_larnv( int idist, int iseed[4], int size, std::complex<double> *x )
{
    lapack_zlarnv( &idist, iseed, &size, x );
}

// -----------------------------------------------------------------------------
#define lapack_slange FORTRAN_NAME( slange, SLANGE )
#define lapack_dlange FORTRAN_NAME( dlange, DLANGE )
#define lapack_clange FORTRAN_NAME( clange, CLANGE )
#define lapack_zlange FORTRAN_NAME( zlange, ZLANGE )

extern "C"
BLAS_RETURN_FLOAT
       lapack_slange( char const *norm,
                      int const *m, int const *n,
                      float const *A, int const *lda,
                      float *work );

extern "C"
double lapack_dlange( char const *norm,
                      int const *m, int const *n,
                      double const *A, int const *lda,
                      double *work );

extern "C"
BLAS_RETURN_FLOAT
       lapack_clange( char const *norm,
                      int const *m, int const *n,
                      std::complex<float> const *A, int const *lda,
                      float *work );

extern "C"
double lapack_zlange( char const *norm,
                      int const *m, int const *n,
                      std::complex<double> const *A, int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lange( char const *norm,
                     int m, int n,
                     float const *A, int lda,
                     float *work )
{
    return lapack_slange( norm, &m, &n, A, &lda, work );
}

inline
double lapack_lange( char const *norm,
                     int m, int n,
                     double const *A, int lda,
                     double *work )
{
    return lapack_dlange( norm, &m, &n, A, &lda, work );
}

inline
float  lapack_lange( char const *norm,
                     int m, int n,
                     std::complex<float> const *A, int lda,
                     float *work )
{
    return lapack_clange( norm, &m, &n, A, &lda, work );
}

inline
double lapack_lange( char const *norm,
                     int m, int n,
                     std::complex<double> const *A, int lda,
                     double *work )
{
    return lapack_zlange( norm, &m, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slansy FORTRAN_NAME( slansy, SLANSY )
#define lapack_dlansy FORTRAN_NAME( dlansy, DLANSY )
#define lapack_clansy FORTRAN_NAME( clansy, CLANSY )
#define lapack_zlansy FORTRAN_NAME( zlansy, ZLANSY )

extern "C"
BLAS_RETURN_FLOAT
       lapack_slansy( char const *norm, char const *uplo,
                      int const *n,
                      float const *A, int const *lda,
                      float *work );

extern "C"
double lapack_dlansy( char const *norm, char const *uplo,
                      int const *n,
                      double const *A, int const *lda,
                      double *work );

extern "C"
BLAS_RETURN_FLOAT
       lapack_clansy( char const *norm, char const *uplo,
                      int const *n,
                      std::complex<float> const *A, int const *lda,
                      float *work );

extern "C"
double lapack_zlansy( char const *norm, char const *uplo,
                      int const *n,
                      std::complex<double> const *A, int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lansy( char const *norm, char const *uplo,
                     int n,
                     float const *A, int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lansy( char const *norm, char const *uplo,
                     int n,
                     double const *A, int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

inline
float  lapack_lansy( char const *norm, char const *uplo,
                     int n,
                     std::complex<float> const *A, int lda,
                     float *work )
{
    return lapack_clansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lansy( char const *norm, char const *uplo,
                     int n,
                     std::complex<double> const *A, int lda,
                     double *work )
{
    return lapack_zlansy( norm, uplo, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_clanhe FORTRAN_NAME( clanhe, CLANHE )
#define lapack_zlanhe FORTRAN_NAME( zlanhe, ZLANHE )

extern "C"
BLAS_RETURN_FLOAT
       lapack_clanhe( char const *norm, char const *uplo,
                      int const *n,
                      std::complex<float> const *A, int const *lda,
                      float *work );

extern "C"
double lapack_zlanhe( char const *norm, char const *uplo,
                      int const *n,
                      std::complex<double> const *A, int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lanhe( char const *norm, char const *uplo,
                     int n,
                     float const *A, int lda,
                     float *work )
{
    return lapack_slansy( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lanhe( char const *norm, char const *uplo,
                     int n,
                     double const *A, int lda,
                     double *work )
{
    return lapack_dlansy( norm, uplo, &n, A, &lda, work );
}

inline
float  lapack_lanhe( char const *norm, char const *uplo,
                     int n,
                     std::complex<float> const *A, int lda,
                     float *work )
{
    return lapack_clanhe( norm, uplo, &n, A, &lda, work );
}

inline
double lapack_lanhe( char const *norm, char const *uplo,
                     int n,
                     std::complex<double> const *A, int lda,
                     double *work )
{
    return lapack_zlanhe( norm, uplo, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slantr FORTRAN_NAME( slantr, SLANTR )
#define lapack_dlantr FORTRAN_NAME( dlantr, DLANTR )
#define lapack_clantr FORTRAN_NAME( clantr, CLANTR )
#define lapack_zlantr FORTRAN_NAME( zlantr, ZLANTR )

extern "C"
BLAS_RETURN_FLOAT
       lapack_slantr( char const *norm, char const *uplo, char const *diag,
                      int const *m, int const *n,
                      float const *A, int const *lda,
                      float *work );

extern "C"
double lapack_dlantr( char const *norm, char const *uplo, char const *diag,
                      int const *m, int const *n,
                      double const *A, int const *lda,
                      double *work );

extern "C"
BLAS_RETURN_FLOAT
       lapack_clantr( char const *norm, char const *uplo, char const *diag,
                      int const *m, int const *n,
                      std::complex<float> const *A, int const *lda,
                      float *work );

extern "C"
double lapack_zlantr( char const *norm, char const *uplo, char const *diag,
                      int const *m, int const *n,
                      std::complex<double> const *A, int const *lda,
                      double *work );

// -----------------------------------------------------------------------------
inline
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int m, int n,
                     float const *A, int lda,
                     float *work )
{
    return lapack_slantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int m, int n,
                     double const *A, int lda,
                     double *work )
{
    return lapack_dlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int m, int n,
                     std::complex<float> const *A, int lda,
                     float *work )
{
    return lapack_clantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

inline
double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int m, int n,
                     std::complex<double> const *A, int lda,
                     double *work )
{
    return lapack_zlantr( norm, uplo, diag, &m, &n, A, &lda, work );
}

// -----------------------------------------------------------------------------
#define lapack_slacpy FORTRAN_NAME( slacpy, SLACPY )
#define lapack_dlacpy FORTRAN_NAME( dlacpy, DLACPY )
#define lapack_clacpy FORTRAN_NAME( clacpy, CLACPY )
#define lapack_zlacpy FORTRAN_NAME( zlacpy, ZLACPY )

extern "C"
void lapack_slacpy( char const* uplo,
                    int const *m, int const *n,
                    float const *A, int const *lda,
                    float const *B, int const *ldb );

extern "C"
void lapack_dlacpy( char const* uplo,
                    int const *m, int const *n,
                    double const *A, int const *lda,
                    double const *B, int const *ldb );

extern "C"
void lapack_clacpy( char const* uplo,
                    int const *m, int const *n,
                    std::complex<float> const *A, int const *lda,
                    std::complex<float> const *B, int const *ldb );

extern "C"
void lapack_zlacpy( char const* uplo,
                    int const *m, int const *n,
                    std::complex<double> const *A, int const *lda,
                    std::complex<double> const *B, int const *ldb );

// -----------------------------------------------------------------------------
inline
void lapack_lacpy( char const* uplo,
                     int m, int n,
                     float const *A, int lda,
                     float const *B, int ldb )
{
    lapack_slacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     int m, int n,
                     double const *A, int lda,
                     double const *B, int ldb )
{
    lapack_dlacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     int m, int n,
                     std::complex<float> const *A, int lda,
                     std::complex<float> const *B, int ldb )
{
    lapack_clacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

inline
void lapack_lacpy( char const* uplo,
                     int m, int n,
                     std::complex<double> const *A, int lda,
                     std::complex<double> const *B, int ldb )
{
    lapack_zlacpy( uplo, &m, &n, A, &lda, B, &ldb );
}

// -----------------------------------------------------------------------------
#define lapack_sgetrf FORTRAN_NAME( sgetrf, SGETRF )
#define lapack_dgetrf FORTRAN_NAME( dgetrf, DGETRF )
#define lapack_cgetrf FORTRAN_NAME( cgetrf, CGETRF )
#define lapack_zgetrf FORTRAN_NAME( zgetrf, ZGETRF )

extern "C"
void lapack_sgetrf( int const *m, int const *n,
                    float *A, int const *lda,
                    int *ipiv, int *info );

extern "C"
void lapack_dgetrf( int const *m, int const *n,
                    double *A, int const *lda,
                    int *ipiv, int *info );

extern "C"
void lapack_cgetrf( int const *m, int const *n,
                    std::complex<float> *A, int const *lda,
                    int *ipiv, int *info );

extern "C"
void lapack_zgetrf( int const *m, int const *n,
                    std::complex<double> *A, int const *lda,
                    int *ipiv, int *info );

// -----------------------------------------------------------------------------
inline
void lapack_getrf(  int m, int n,
                    float *A, int lda,
                    int *ipiv, int *info )
{
    lapack_sgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  int m, int n,
                    double *A, int lda,
                    int *ipiv, int *info )
{
    lapack_dgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  int m, int n,
                    std::complex<float> *A, int lda,
                    int *ipiv, int *info )
{
    lapack_cgetrf( &m, &n, A, &lda, ipiv, info );
}

inline
void lapack_getrf(  int m, int n,
                    std::complex<double> *A, int lda,
                    int *ipiv, int *info )
{
    lapack_zgetrf( &m, &n, A, &lda, ipiv, info );
}

#endif        //  #ifndef LAPACK_HH
