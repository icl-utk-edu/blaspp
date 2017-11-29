#ifndef BLAS_FORTRAN_HH
#define BLAS_FORTRAN_HH

#include "blas_mangling.hh"

#include <complex>

// -----------------------------------------------------------------------------
// blas_int is the integer type of the underlying Fortran BLAS library.
// BLAS wrappers take int64_t and check for overflow before casting to blas_int.
#ifdef BLAS_ILP64
    typedef long long blas_int;
#else
    typedef int blas_int;
#endif

// -----------------------------------------------------------------------------
// f2c, hence MacOS Accelerate, returns double instead of float for sdot, etc.
#if defined(HAVE_MACOS_ACCELERATE) || defined(HAVE_F2C)
    typedef double blas_float_return;
#else
    typedef float blas_float_return;
#endif

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Level 1 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define f77_saxpy FORTRAN_NAME( saxpy, SAXPY )
void f77_saxpy(
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float       *y, blas_int const *incy );

#define f77_daxpy FORTRAN_NAME( daxpy, DAXPY )
void f77_daxpy(
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double       *y, blas_int const *incy );

#define f77_caxpy FORTRAN_NAME( caxpy, CAXPY )
void f77_caxpy(
    blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float>       *y, blas_int const *incy );

#define f77_zaxpy FORTRAN_NAME( zaxpy, ZAXPY )
void f77_zaxpy(
    blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double>       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_sscal FORTRAN_NAME( sscal, SSCAL )
void f77_sscal(
    blas_int const *n,
    float const *alpha,
    float       *x, blas_int const *incx );

#define f77_dscal FORTRAN_NAME( dscal, DSCAL )
void f77_dscal(
    blas_int const *n,
    double const *alpha,
    double       *x, blas_int const *incx );

#define f77_cscal FORTRAN_NAME( cscal, CSCAL )
void f77_cscal(
    blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float>       *x, blas_int const *incx );

#define f77_zscal FORTRAN_NAME( zscal, ZSCAL )
void f77_zscal(
    blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double>       *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define f77_scopy FORTRAN_NAME( scopy, SCOPY )
void f77_scopy(
    blas_int const *n,
    float const *x, blas_int const *incx,
    float       *y, blas_int const *incy );

#define f77_dcopy FORTRAN_NAME( dcopy, DCOPY )
void f77_dcopy(
    blas_int const *n,
    double const *x, blas_int const *incx,
    double       *y, blas_int const *incy );

#define f77_ccopy FORTRAN_NAME( ccopy, CCOPY )
void f77_ccopy(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float>       *y, blas_int const *incy );

#define f77_zcopy FORTRAN_NAME( zcopy, ZCOPY )
void f77_zcopy(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double>       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_sswap FORTRAN_NAME( sswap, SSWAP )
void f77_sswap(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy );

#define f77_dswap FORTRAN_NAME( dswap, DSWAP )
void f77_dswap(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy );

#define f77_cswap FORTRAN_NAME( cswap, CSWAP )
void f77_cswap(
    blas_int const *n,
    std::complex<float> *x, blas_int const *incx,
    std::complex<float> *y, blas_int const *incy );

#define f77_zswap FORTRAN_NAME( zswap, ZSWAP )
void f77_zswap(
    blas_int const *n,
    std::complex<double> *x, blas_int const *incx,
    std::complex<double> *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_sdot FORTRAN_NAME( sdot, SDOT )
blas_float_return f77_sdot(
    blas_int const *n,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy );

#define f77_ddot FORTRAN_NAME( ddot, DDOT )
double f77_ddot(
    blas_int const *n,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy );

// -----------------------------------------------------------------------------
// For Fortran functions returning complex values,
// define BLAS_COMPLEX_RETURN_ARGUMENT if result is a hidden first argument (Intel icc),
// else the default is to return complex values (GNU gcc).
#ifdef BLAS_COMPLEX_RETURN_ARGUMENT

#define f77_cdotc FORTRAN_NAME( cdotc, CDOTC )
void f77_cdotc(
    std::complex<float> *result,
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy );

#define f77_zdotc FORTRAN_NAME( zdotc, ZDOTC )
void f77_zdotc(
    std::complex<double> *result,
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy );

#define f77_cdotu FORTRAN_NAME( cdotu, CDOTU )
void f77_cdotu(
    std::complex<float> *result,
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy );

#define f77_zdotu FORTRAN_NAME( zdotu, ZDOTU )
void f77_zdotu(
    std::complex<double> *result,
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy );

// --------------------
#else // ! defined(BLAS_COMPLEX_RETURN_ARGUMENT)

#define f77_cdotc FORTRAN_NAME( cdotc, CDOTC )
std::complex<float> f77_cdotc(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy );

#define f77_zdotc FORTRAN_NAME( zdotc, ZDOTC )
std::complex<double> f77_zdotc(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy );

#define f77_cdotu FORTRAN_NAME( cdotu, CDOTU )
std::complex<float> f77_cdotu(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy );

#define f77_zdotu FORTRAN_NAME( zdotu, ZDOTU )
std::complex<double> f77_zdotu(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy );

#endif // ! defined(BLAS_COMPLEX_RETURN)

// -----------------------------------------------------------------------------
#define f77_snrm2 FORTRAN_NAME( snrm2, SNRM2 )
blas_float_return f77_snrm2(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define f77_dnrm2 FORTRAN_NAME( dnrm2, DNRM2 )
double f77_dnrm2(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define f77_scnrm2 FORTRAN_NAME( scnrm2, SCNRM2 )
blas_float_return f77_scnrm2(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx );

#define f77_dznrm2 FORTRAN_NAME( dznrm2, DZNRM2 )
double f77_dznrm2(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define f77_sasum FORTRAN_NAME( sasum, SASUM )
blas_float_return f77_sasum(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define f77_dasum FORTRAN_NAME( dasum, DASUM )
double f77_dasum(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define f77_scasum FORTRAN_NAME( scasum, SCASUM )
blas_float_return f77_scasum(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx );

#define f77_dzasum FORTRAN_NAME( dzasum, DZASUM )
double f77_dzasum(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define f77_isamax FORTRAN_NAME( isamax, ISAMAX )
blas_int f77_isamax(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define f77_idamax FORTRAN_NAME( idamax, IDAMAX )
blas_int f77_idamax(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define f77_icamax FORTRAN_NAME( icamax, ICAMAX )
blas_int f77_icamax(
    blas_int const *n,
    std::complex<float> const *x, blas_int const *incx );

#define f77_izamax FORTRAN_NAME( izamax, IZAMAX )
blas_int f77_izamax(
    blas_int const *n,
    std::complex<double> const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
// c is real
// oddly, b is const for crotg, zrotg
#define f77_srotg FORTRAN_NAME( srotg, SROTG )
void f77_srotg(
    float *a,
    float *b,
    float *c,
    float *s );

#define f77_drotg FORTRAN_NAME( drotg, DROTG )
void f77_drotg(
    double *a,
    double *b,
    double *c,
    double *s );

#define f77_crotg FORTRAN_NAME( crotg, CROTG )
void f77_crotg(
    std::complex<float> *a,
    std::complex<float> const *b,
    float *c,
    std::complex<float> *s );

#define f77_zrotg FORTRAN_NAME( zrotg, ZROTG )
void f77_zrotg(
    std::complex<double> *a,
    std::complex<double> const *b,
    double *c,
    std::complex<double> *s );

// -----------------------------------------------------------------------------
// c is real
#define f77_srot FORTRAN_NAME( srot, SROT )
void f77_srot(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy,
    float const *c,
    float const *s );

#define f77_drot FORTRAN_NAME( drot, DROT )
void f77_drot(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy,
    double const *c,
    double const *s );

#define f77_csrot FORTRAN_NAME( csrot, CSROT )
void f77_csrot(
    blas_int const *n,
    std::complex<float> *x, blas_int const *incx,
    std::complex<float> *y, blas_int const *incy,
    float const *c,
    float const *s );

#define f77_zdrot FORTRAN_NAME( zdrot, ZDROT )
void f77_zdrot(
    blas_int const *n,
    std::complex<double> *x, blas_int const *incx,
    std::complex<double> *y, blas_int const *incy,
    double const *c,
    double const *s );

#define f77_crot FORTRAN_NAME( crot, CROT )
void f77_crot(
    blas_int const *n,
    std::complex<float> *x, blas_int const *incx,
    std::complex<float> *y, blas_int const *incy,
    float const *c,
    std::complex<float> const *s );

#define f77_zrot FORTRAN_NAME( zrot, ZROT )
void f77_zrot(
    blas_int const *n,
    std::complex<double> *x, blas_int const *incx,
    std::complex<double> *y, blas_int const *incy,
    double const *c,
    std::complex<double> const *s );

// -----------------------------------------------------------------------------
#define f77_srotmg FORTRAN_NAME( srotmg, SROTMG )
void f77_srotmg(
    float *d1,
    float *d2,
    float *x1,
    float const *y1,
    float *param );

#define f77_drotmg FORTRAN_NAME( drotmg, DROTMG )
void f77_drotmg(
    double *d1,
    double *d2,
    double *x1,
    double const *y1,
    double *param );

#define f77_crotmg FORTRAN_NAME( crotmg, CROTMG )
void f77_crotmg(
    std::complex<float> *d1,
    std::complex<float> *d2,
    std::complex<float> *x1,
    std::complex<float> const *y1,
    std::complex<float> *param );

#define f77_zrotmg FORTRAN_NAME( zrotmg, ZROTMG )
void f77_zrotmg(
    std::complex<double> *d1,
    std::complex<double> *d2,
    std::complex<double> *x1,
    std::complex<double> const *y1,
    std::complex<double> *param );

// -----------------------------------------------------------------------------
#define f77_srotm FORTRAN_NAME( srotm, SROTM )
void f77_srotm(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy,
    float const *param );

#define f77_drotm FORTRAN_NAME( drotm, DROTM )
void f77_drotm(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy,
    double const *param );

#define f77_crotm FORTRAN_NAME( crotm, CROTM )
void f77_crotm(
    blas_int const *n,
    std::complex<float> *x, blas_int const *incx,
    std::complex<float> *y, blas_int const *incy,
    std::complex<float> const *param );

#define f77_zrotm FORTRAN_NAME( zrotm, ZROTM )
void f77_zrotm(
    blas_int const *n,
    std::complex<double> *x, blas_int const *incx,
    std::complex<double> *y, blas_int const *incy,
    std::complex<double> const *param );

// =============================================================================
// Level 2 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define f77_sgemv FORTRAN_NAME( sgemv, SGEMV )
void f77_sgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *x, blas_int const *incx,
    float const *beta,
    float       *y, blas_int const *incy );

#define f77_dgemv FORTRAN_NAME( dgemv, DGEMV )
void f77_dgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *x, blas_int const *incx,
    double const *beta,
    double       *y, blas_int const *incy );

#define f77_cgemv FORTRAN_NAME( cgemv, CGEMV )
void f77_cgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *beta,
    std::complex<float>       *y, blas_int const *incy );

#define f77_zgemv FORTRAN_NAME( zgemv, ZGEMV )
void f77_zgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *beta,
    std::complex<double>       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_sger FORTRAN_NAME( sger, SGER )
void f77_sger(
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy,
    float       *A, blas_int const *lda );

#define f77_dger FORTRAN_NAME( dger, DGER )
void f77_dger(
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy,
    double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define f77_cgerc FORTRAN_NAME( cgerc, CGERC )
void f77_cgerc(
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy,
    std::complex<float>       *A, blas_int const *lda );

#define f77_zgerc FORTRAN_NAME( zgerc, ZGERC )
void f77_zgerc(
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy,
    std::complex<double>       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define f77_cgeru FORTRAN_NAME( cgeru, CGERU )
void f77_cgeru(
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy,
    std::complex<float>       *A, blas_int const *lda );

#define f77_zgeru FORTRAN_NAME( zgeru, ZGERU )
void f77_zgeru(
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy,
    std::complex<double>       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define f77_ssymv FORTRAN_NAME( ssymv, SSYMV )
void f77_ssymv(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *x, blas_int const *incx,
    float const *beta,
    float       *y, blas_int const *incy );

#define f77_dsymv FORTRAN_NAME( dsymv, DSYMV )
void f77_dsymv(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *x, blas_int const *incx,
    double const *beta,
    double       *y, blas_int const *incy );

#define f77_csymv FORTRAN_NAME( csymv, CSYMV )
void f77_csymv(
    char const *uplo,
    blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *beta,
    std::complex<float>       *y, blas_int const *incy );

#define f77_zsymv FORTRAN_NAME( zsymv, ZSYMV )
void f77_zsymv(
    char const *uplo,
    blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *beta,
    std::complex<double>       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_chemv FORTRAN_NAME( chemv, CHEMV )
void f77_chemv(
    char const *uplo,
    blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *beta,
    std::complex<float>       *y, blas_int const *incy );

#define f77_zhemv FORTRAN_NAME( zhemv, ZHEMV )
void f77_zhemv(
    char const *uplo,
    blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *beta,
    std::complex<double>       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define f77_ssyr FORTRAN_NAME( ssyr, SSYR )
void f77_ssyr(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float       *A, blas_int const *lda );

#define f77_dsyr FORTRAN_NAME( dsyr, DSYR )
void f77_dsyr(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double       *A, blas_int const *lda );

// conflicts with current prototype in lapacke.h
//#define f77_csyr FORTRAN_NAME( csyr, CSYR )
//void FORTRAN_NAME( csyr, CSYR )(
//    char const *uplo,
//    blas_int const *n,
//    std::complex<float> const *alpha,
//    std::complex<float> const *x, blas_int const *incx,
//    std::complex<float>       *A, blas_int const *lda );
//
//#define f77_zsyr FORTRAN_NAME( zsyr, ZSYR )
//void f77_zsyr(
//    char const *uplo,
//    blas_int const *n,
//    std::complex<double> const *alpha,
//    std::complex<double> const *x, blas_int const *incx,
//    std::complex<double>       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
// alpha is real
#define f77_cher FORTRAN_NAME( cher, CHER )
void f77_cher(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float>       *A, blas_int const *lda );

#define f77_zher FORTRAN_NAME( zher, ZHER )
void f77_zher(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double>       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
// [cz]syr2 not available in standard BLAS or LAPACK; use [cz]syr2k with k=1.
#define f77_ssyr2 FORTRAN_NAME( ssyr2, SSYR2 )
void f77_ssyr2(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy,
    float       *A, blas_int const *lda );

#define f77_dsyr2 FORTRAN_NAME( dsyr2, DSYR2 )
void f77_dsyr2(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy,
    double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define f77_cher2 FORTRAN_NAME( cher2, CHER2 )
void f77_cher2(
    char const *uplo,
    blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *x, blas_int const *incx,
    std::complex<float> const *y, blas_int const *incy,
    std::complex<float>       *A, blas_int const *lda );

#define f77_zher2 FORTRAN_NAME( zher2, ZHER2 )
void f77_zher2(
    char const *uplo,
    blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *x, blas_int const *incx,
    std::complex<double> const *y, blas_int const *incy,
    std::complex<double>       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define f77_strmv FORTRAN_NAME( strmv, STRMV )
void f77_strmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    float const *A, blas_int const *lda,
    float       *x, blas_int const *incx );

#define f77_dtrmv FORTRAN_NAME( dtrmv, DTRMV )
void f77_dtrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    double const *A, blas_int const *lda,
    double       *x, blas_int const *incx );

#define f77_ctrmv FORTRAN_NAME( ctrmv, CTRMV )
void f77_ctrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float>       *x, blas_int const *incx );

#define f77_ztrmv FORTRAN_NAME( ztrmv, ZTRMV )
void f77_ztrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double>       *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define f77_strsv FORTRAN_NAME( strsv, STRSV )
void f77_strsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    float const *A, blas_int const *lda,
    float       *x, blas_int const *incx );

#define f77_dtrsv FORTRAN_NAME( dtrsv, DTRSV )
void f77_dtrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    double const *A, blas_int const *lda,
    double       *x, blas_int const *incx );

#define f77_ctrsv FORTRAN_NAME( ctrsv, CTRSV )
void f77_ctrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float>       *x, blas_int const *incx );

#define f77_ztrsv FORTRAN_NAME( ztrsv, ZTRSV )
void f77_ztrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double>       *x, blas_int const *incx );

// =============================================================================
// Level 3 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define f77_sgemm FORTRAN_NAME( sgemm, SGEMM )
void f77_sgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define f77_dgemm FORTRAN_NAME( dgemm, DGEMM )
void f77_dgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define f77_cgemm FORTRAN_NAME( cgemm, CGEMM )
void f77_cgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *B, blas_int const *ldb,
    std::complex<float> const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zgemm FORTRAN_NAME( zgemm, ZGEMM )
void f77_zgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *B, blas_int const *ldb,
    std::complex<double> const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define f77_ssymm FORTRAN_NAME( ssymm, SSYMM )
void f77_ssymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define f77_dsymm FORTRAN_NAME( dsymm, DSYMM )
void f77_dsymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define f77_csymm FORTRAN_NAME( csymm, CSYMM )
void f77_csymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *B, blas_int const *ldb,
    std::complex<float> const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zsymm FORTRAN_NAME( zsymm, ZSYMM )
void f77_zsymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *B, blas_int const *ldb,
    std::complex<double> const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define f77_chemm FORTRAN_NAME( chemm, CHEMM )
void f77_chemm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *B, blas_int const *ldb,
    std::complex<float> const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zhemm FORTRAN_NAME( zhemm, ZHEMM )
void f77_zhemm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *B, blas_int const *ldb,
    std::complex<double> const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define f77_ssyrk FORTRAN_NAME( ssyrk, SSYRK )
void f77_ssyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *beta,
    float       *C, blas_int const *ldc );

#define f77_dsyrk FORTRAN_NAME( dsyrk, DSYRK )
void f77_dsyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *beta,
    double       *C, blas_int const *ldc );

#define f77_csyrk FORTRAN_NAME( csyrk, CSYRK )
void f77_csyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zsyrk FORTRAN_NAME( zsyrk, ZSYRK )
void f77_zsyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
// alpha and beta are real
#define f77_cherk FORTRAN_NAME( cherk, CHERK )
void f77_cherk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    float const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zherk FORTRAN_NAME( zherk, ZHERK )
void f77_zherk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    double const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define f77_ssyr2k FORTRAN_NAME( ssyr2k, SSYR2K )
void f77_ssyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define f77_dsyr2k FORTRAN_NAME( dsyr2k, DSYR2K )
void f77_dsyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define f77_csyr2k FORTRAN_NAME( csyr2k, CSYR2K )
void f77_csyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *B, blas_int const *ldb,
    std::complex<float> const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zsyr2k FORTRAN_NAME( zsyr2k, ZSYR2K )
void f77_zsyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *B, blas_int const *ldb,
    std::complex<double> const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
// beta is real
#define f77_cher2k FORTRAN_NAME( cher2k, CHER2K )
void f77_cher2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float> const *B, blas_int const *ldb,
    float const *beta,
    std::complex<float>       *C, blas_int const *ldc );

#define f77_zher2k FORTRAN_NAME( zher2k, ZHER2K )
void f77_zher2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double> const *B, blas_int const *ldb,
    double const *beta,
    std::complex<double>       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define f77_strmm FORTRAN_NAME( strmm, STRMM )
void f77_strmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float       *B, blas_int const *ldb );

#define f77_dtrmm FORTRAN_NAME( dtrmm, DTRMM )
void f77_dtrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double       *B, blas_int const *ldb );

#define f77_ctrmm FORTRAN_NAME( ctrmm, CTRMM )
void f77_ctrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float>       *B, blas_int const *ldb );

#define f77_ztrmm FORTRAN_NAME( ztrmm, ZTRMM )
void f77_ztrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double>       *B, blas_int const *ldb );

// -----------------------------------------------------------------------------
#define f77_strsm FORTRAN_NAME( strsm, STRSM )
void f77_strsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float       *B, blas_int const *ldb );

#define f77_dtrsm FORTRAN_NAME( dtrsm, DTRSM )
void f77_dtrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double       *B, blas_int const *ldb );

#define f77_ctrsm FORTRAN_NAME( ctrsm, CTRSM )
void f77_ctrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    std::complex<float> const *alpha,
    std::complex<float> const *A, blas_int const *lda,
    std::complex<float>       *B, blas_int const *ldb );

#define f77_ztrsm FORTRAN_NAME( ztrsm, ZTRSM )
void f77_ztrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    std::complex<double> const *alpha,
    std::complex<double> const *A, blas_int const *lda,
    std::complex<double>       *B, blas_int const *ldb );

#ifdef __cplusplus
}  // #endif
#endif

#endif        //  #ifndef BLAS_FORTRAN_HH
