// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_FORTRAN_H
#define BLAS_FORTRAN_H

#include "blas/mangling.h"
#include "blas/config.h"

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Level 1 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define BLAS_saxpy BLAS_FORTRAN_NAME( saxpy, SAXPY )
void BLAS_saxpy(
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float       *y, blas_int const *incy );

#define BLAS_daxpy BLAS_FORTRAN_NAME( daxpy, DAXPY )
void BLAS_daxpy(
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double       *y, blas_int const *incy );

#define BLAS_caxpy BLAS_FORTRAN_NAME( caxpy, CAXPY )
void BLAS_caxpy(
    blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float       *y, blas_int const *incy );

#define BLAS_zaxpy BLAS_FORTRAN_NAME( zaxpy, ZAXPY )
void BLAS_zaxpy(
    blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_sscal BLAS_FORTRAN_NAME( sscal, SSCAL )
void BLAS_sscal(
    blas_int const *n,
    float const *alpha,
    float       *x, blas_int const *incx );

#define BLAS_dscal BLAS_FORTRAN_NAME( dscal, DSCAL )
void BLAS_dscal(
    blas_int const *n,
    double const *alpha,
    double       *x, blas_int const *incx );

#define BLAS_cscal BLAS_FORTRAN_NAME( cscal, CSCAL )
void BLAS_cscal(
    blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float       *x, blas_int const *incx );

#define BLAS_zscal BLAS_FORTRAN_NAME( zscal, ZSCAL )
void BLAS_zscal(
    blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double       *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define BLAS_scopy BLAS_FORTRAN_NAME( scopy, SCOPY )
void BLAS_scopy(
    blas_int const *n,
    float const *x, blas_int const *incx,
    float       *y, blas_int const *incy );

#define BLAS_dcopy BLAS_FORTRAN_NAME( dcopy, DCOPY )
void BLAS_dcopy(
    blas_int const *n,
    double const *x, blas_int const *incx,
    double       *y, blas_int const *incy );

#define BLAS_ccopy BLAS_FORTRAN_NAME( ccopy, CCOPY )
void BLAS_ccopy(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float       *y, blas_int const *incy );

#define BLAS_zcopy BLAS_FORTRAN_NAME( zcopy, ZCOPY )
void BLAS_zcopy(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_sswap BLAS_FORTRAN_NAME( sswap, SSWAP )
void BLAS_sswap(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy );

#define BLAS_dswap BLAS_FORTRAN_NAME( dswap, DSWAP )
void BLAS_dswap(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy );

#define BLAS_cswap BLAS_FORTRAN_NAME( cswap, CSWAP )
void BLAS_cswap(
    blas_int const *n,
    blas_complex_float *x, blas_int const *incx,
    blas_complex_float *y, blas_int const *incy );

#define BLAS_zswap BLAS_FORTRAN_NAME( zswap, ZSWAP )
void BLAS_zswap(
    blas_int const *n,
    blas_complex_double *x, blas_int const *incx,
    blas_complex_double *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_sdot BLAS_FORTRAN_NAME( sdot, SDOT )
blas_float_return BLAS_sdot(
    blas_int const *n,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy );

#define BLAS_ddot BLAS_FORTRAN_NAME( ddot, DDOT )
double BLAS_ddot(
    blas_int const *n,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy );

// -----------------------------------------------------------------------------
// For Fortran functions returning complex values,
// define BLAS_COMPLEX_RETURN_ARGUMENT if result is a hidden first argument (Intel icc),
// else the default is to return complex values (GNU gcc).
#ifdef BLAS_COMPLEX_RETURN_ARGUMENT

#define BLAS_cdotc BLAS_FORTRAN_NAME( cdotc, CDOTC )
void BLAS_cdotc(
    blas_complex_float *result,
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy );

#define BLAS_zdotc BLAS_FORTRAN_NAME( zdotc, ZDOTC )
void BLAS_zdotc(
    blas_complex_double *result,
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy );

#define BLAS_cdotu BLAS_FORTRAN_NAME( cdotu, CDOTU )
void BLAS_cdotu(
    blas_complex_float *result,
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy );

#define BLAS_zdotu BLAS_FORTRAN_NAME( zdotu, ZDOTU )
void BLAS_zdotu(
    blas_complex_double *result,
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy );

// --------------------
#else // ! defined(BLAS_COMPLEX_RETURN_ARGUMENT)

#define BLAS_cdotc BLAS_FORTRAN_NAME( cdotc, CDOTC )
blas_complex_float BLAS_cdotc(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy );

#define BLAS_zdotc BLAS_FORTRAN_NAME( zdotc, ZDOTC )
blas_complex_double BLAS_zdotc(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy );

#define BLAS_cdotu BLAS_FORTRAN_NAME( cdotu, CDOTU )
blas_complex_float BLAS_cdotu(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy );

#define BLAS_zdotu BLAS_FORTRAN_NAME( zdotu, ZDOTU )
blas_complex_double BLAS_zdotu(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy );

#endif // ! defined(BLAS_COMPLEX_RETURN)

// -----------------------------------------------------------------------------
#define BLAS_snrm2 BLAS_FORTRAN_NAME( snrm2, SNRM2 )
blas_float_return BLAS_snrm2(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define BLAS_dnrm2 BLAS_FORTRAN_NAME( dnrm2, DNRM2 )
double BLAS_dnrm2(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define BLAS_scnrm2 BLAS_FORTRAN_NAME( scnrm2, SCNRM2 )
blas_float_return BLAS_scnrm2(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx );

#define BLAS_dznrm2 BLAS_FORTRAN_NAME( dznrm2, DZNRM2 )
double BLAS_dznrm2(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define BLAS_sasum BLAS_FORTRAN_NAME( sasum, SASUM )
blas_float_return BLAS_sasum(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define BLAS_dasum BLAS_FORTRAN_NAME( dasum, DASUM )
double BLAS_dasum(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define BLAS_scasum BLAS_FORTRAN_NAME( scasum, SCASUM )
blas_float_return BLAS_scasum(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx );

#define BLAS_dzasum BLAS_FORTRAN_NAME( dzasum, DZASUM )
double BLAS_dzasum(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define BLAS_isamax BLAS_FORTRAN_NAME( isamax, ISAMAX )
blas_int BLAS_isamax(
    blas_int const *n,
    float const *x, blas_int const *incx );

#define BLAS_idamax BLAS_FORTRAN_NAME( idamax, IDAMAX )
blas_int BLAS_idamax(
    blas_int const *n,
    double const *x, blas_int const *incx );

#define BLAS_icamax BLAS_FORTRAN_NAME( icamax, ICAMAX )
blas_int BLAS_icamax(
    blas_int const *n,
    blas_complex_float const *x, blas_int const *incx );

#define BLAS_izamax BLAS_FORTRAN_NAME( izamax, IZAMAX )
blas_int BLAS_izamax(
    blas_int const *n,
    blas_complex_double const *x, blas_int const *incx );

// -----------------------------------------------------------------------------
// c is real
// oddly, b is const for crotg, zrotg
#define BLAS_srotg BLAS_FORTRAN_NAME( srotg, SROTG )
void BLAS_srotg(
    float *a,
    float *b,
    float *c,
    float *s );

#define BLAS_drotg BLAS_FORTRAN_NAME( drotg, DROTG )
void BLAS_drotg(
    double *a,
    double *b,
    double *c,
    double *s );

#define BLAS_crotg BLAS_FORTRAN_NAME( crotg, CROTG )
void BLAS_crotg(
    blas_complex_float *a,
    blas_complex_float const *b,
    float *c,
    blas_complex_float *s );

#define BLAS_zrotg BLAS_FORTRAN_NAME( zrotg, ZROTG )
void BLAS_zrotg(
    blas_complex_double *a,
    blas_complex_double const *b,
    double *c,
    blas_complex_double *s );

// -----------------------------------------------------------------------------
// c is real
#define BLAS_srot BLAS_FORTRAN_NAME( srot, SROT )
void BLAS_srot(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy,
    float const *c,
    float const *s );

#define BLAS_drot BLAS_FORTRAN_NAME( drot, DROT )
void BLAS_drot(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy,
    double const *c,
    double const *s );

#define BLAS_csrot BLAS_FORTRAN_NAME( csrot, CSROT )
void BLAS_csrot(
    blas_int const *n,
    blas_complex_float *x, blas_int const *incx,
    blas_complex_float *y, blas_int const *incy,
    float const *c,
    float const *s );

#define BLAS_zdrot BLAS_FORTRAN_NAME( zdrot, ZDROT )
void BLAS_zdrot(
    blas_int const *n,
    blas_complex_double *x, blas_int const *incx,
    blas_complex_double *y, blas_int const *incy,
    double const *c,
    double const *s );

#define BLAS_crot BLAS_FORTRAN_NAME( crot, CROT )
void BLAS_crot(
    blas_int const *n,
    blas_complex_float *x, blas_int const *incx,
    blas_complex_float *y, blas_int const *incy,
    float const *c,
    blas_complex_float const *s );

#define BLAS_zrot BLAS_FORTRAN_NAME( zrot, ZROT )
void BLAS_zrot(
    blas_int const *n,
    blas_complex_double *x, blas_int const *incx,
    blas_complex_double *y, blas_int const *incy,
    double const *c,
    blas_complex_double const *s );

// -----------------------------------------------------------------------------
#define BLAS_srotmg BLAS_FORTRAN_NAME( srotmg, SROTMG )
void BLAS_srotmg(
    float *d1,
    float *d2,
    float *x1,
    float const *y1,
    float *param );

#define BLAS_drotmg BLAS_FORTRAN_NAME( drotmg, DROTMG )
void BLAS_drotmg(
    double *d1,
    double *d2,
    double *x1,
    double const *y1,
    double *param );

#define BLAS_crotmg BLAS_FORTRAN_NAME( crotmg, CROTMG )
void BLAS_crotmg(
    blas_complex_float *d1,
    blas_complex_float *d2,
    blas_complex_float *x1,
    blas_complex_float const *y1,
    blas_complex_float *param );

#define BLAS_zrotmg BLAS_FORTRAN_NAME( zrotmg, ZROTMG )
void BLAS_zrotmg(
    blas_complex_double *d1,
    blas_complex_double *d2,
    blas_complex_double *x1,
    blas_complex_double const *y1,
    blas_complex_double *param );

// -----------------------------------------------------------------------------
#define BLAS_srotm BLAS_FORTRAN_NAME( srotm, SROTM )
void BLAS_srotm(
    blas_int const *n,
    float *x, blas_int const *incx,
    float *y, blas_int const *incy,
    float const *param );

#define BLAS_drotm BLAS_FORTRAN_NAME( drotm, DROTM )
void BLAS_drotm(
    blas_int const *n,
    double *x, blas_int const *incx,
    double *y, blas_int const *incy,
    double const *param );

#define BLAS_crotm BLAS_FORTRAN_NAME( crotm, CROTM )
void BLAS_crotm(
    blas_int const *n,
    blas_complex_float *x, blas_int const *incx,
    blas_complex_float *y, blas_int const *incy,
    blas_complex_float const *param );

#define BLAS_zrotm BLAS_FORTRAN_NAME( zrotm, ZROTM )
void BLAS_zrotm(
    blas_int const *n,
    blas_complex_double *x, blas_int const *incx,
    blas_complex_double *y, blas_int const *incy,
    blas_complex_double const *param );

// =============================================================================
// Level 2 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define BLAS_sgemv BLAS_FORTRAN_NAME( sgemv, SGEMV )
void BLAS_sgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *x, blas_int const *incx,
    float const *beta,
    float       *y, blas_int const *incy );

#define BLAS_dgemv BLAS_FORTRAN_NAME( dgemv, DGEMV )
void BLAS_dgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *x, blas_int const *incx,
    double const *beta,
    double       *y, blas_int const *incy );

#define BLAS_cgemv BLAS_FORTRAN_NAME( cgemv, CGEMV )
void BLAS_cgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *beta,
    blas_complex_float       *y, blas_int const *incy );

#define BLAS_zgemv BLAS_FORTRAN_NAME( zgemv, ZGEMV )
void BLAS_zgemv(
    char const *trans,
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *beta,
    blas_complex_double       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_sger BLAS_FORTRAN_NAME( sger, SGER )
void BLAS_sger(
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy,
    float       *A, blas_int const *lda );

#define BLAS_dger BLAS_FORTRAN_NAME( dger, DGER )
void BLAS_dger(
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy,
    double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define BLAS_cgerc BLAS_FORTRAN_NAME( cgerc, CGERC )
void BLAS_cgerc(
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy,
    blas_complex_float       *A, blas_int const *lda );

#define BLAS_zgerc BLAS_FORTRAN_NAME( zgerc, ZGERC )
void BLAS_zgerc(
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy,
    blas_complex_double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define BLAS_cgeru BLAS_FORTRAN_NAME( cgeru, CGERU )
void BLAS_cgeru(
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy,
    blas_complex_float       *A, blas_int const *lda );

#define BLAS_zgeru BLAS_FORTRAN_NAME( zgeru, ZGERU )
void BLAS_zgeru(
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy,
    blas_complex_double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define BLAS_ssymv BLAS_FORTRAN_NAME( ssymv, SSYMV )
void BLAS_ssymv(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *x, blas_int const *incx,
    float const *beta,
    float       *y, blas_int const *incy );

#define BLAS_dsymv BLAS_FORTRAN_NAME( dsymv, DSYMV )
void BLAS_dsymv(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *x, blas_int const *incx,
    double const *beta,
    double       *y, blas_int const *incy );

// [cz]symv moved to LAPACK++ since they are provided by LAPACK.
// #define BLAS_csymv BLAS_FORTRAN_NAME( csymv, CSYMV )
// void BLAS_csymv(
//     char const *uplo,
//     blas_int const *n,
//     blas_complex_float const *alpha,
//     blas_complex_float const *A, blas_int const *lda,
//     blas_complex_float const *x, blas_int const *incx,
//     blas_complex_float const *beta,
//     blas_complex_float       *y, blas_int const *incy );
//
// #define BLAS_zsymv BLAS_FORTRAN_NAME( zsymv, ZSYMV )
// void BLAS_zsymv(
//     char const *uplo,
//     blas_int const *n,
//     blas_complex_double const *alpha,
//     blas_complex_double const *A, blas_int const *lda,
//     blas_complex_double const *x, blas_int const *incx,
//     blas_complex_double const *beta,
//     blas_complex_double       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_chemv BLAS_FORTRAN_NAME( chemv, CHEMV )
void BLAS_chemv(
    char const *uplo,
    blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *beta,
    blas_complex_float       *y, blas_int const *incy );

#define BLAS_zhemv BLAS_FORTRAN_NAME( zhemv, ZHEMV )
void BLAS_zhemv(
    char const *uplo,
    blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *beta,
    blas_complex_double       *y, blas_int const *incy );

// -----------------------------------------------------------------------------
#define BLAS_ssyr BLAS_FORTRAN_NAME( ssyr, SSYR )
void BLAS_ssyr(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float       *A, blas_int const *lda );

#define BLAS_dsyr BLAS_FORTRAN_NAME( dsyr, DSYR )
void BLAS_dsyr(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double       *A, blas_int const *lda );

// conflicts with current prototype in lapacke.h
//#define BLAS_csyr BLAS_FORTRAN_NAME( csyr, CSYR )
//void BLAS_FORTRAN_NAME( csyr, CSYR )(
//    char const *uplo,
//    blas_int const *n,
//    blas_complex_float const *alpha,
//    blas_complex_float const *x, blas_int const *incx,
//    blas_complex_float       *A, blas_int const *lda );
//
//#define BLAS_zsyr BLAS_FORTRAN_NAME( zsyr, ZSYR )
//void BLAS_zsyr(
//    char const *uplo,
//    blas_int const *n,
//    blas_complex_double const *alpha,
//    blas_complex_double const *x, blas_int const *incx,
//    blas_complex_double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
// alpha is real
#define BLAS_cher BLAS_FORTRAN_NAME( cher, CHER )
void BLAS_cher(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float       *A, blas_int const *lda );

#define BLAS_zher BLAS_FORTRAN_NAME( zher, ZHER )
void BLAS_zher(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
// [cz]syr2 not available in standard BLAS or LAPACK; use [cz]syr2k with k=1.
#define BLAS_ssyr2 BLAS_FORTRAN_NAME( ssyr2, SSYR2 )
void BLAS_ssyr2(
    char const *uplo,
    blas_int const *n,
    float const *alpha,
    float const *x, blas_int const *incx,
    float const *y, blas_int const *incy,
    float       *A, blas_int const *lda );

#define BLAS_dsyr2 BLAS_FORTRAN_NAME( dsyr2, DSYR2 )
void BLAS_dsyr2(
    char const *uplo,
    blas_int const *n,
    double const *alpha,
    double const *x, blas_int const *incx,
    double const *y, blas_int const *incy,
    double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define BLAS_cher2 BLAS_FORTRAN_NAME( cher2, CHER2 )
void BLAS_cher2(
    char const *uplo,
    blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *x, blas_int const *incx,
    blas_complex_float const *y, blas_int const *incy,
    blas_complex_float       *A, blas_int const *lda );

#define BLAS_zher2 BLAS_FORTRAN_NAME( zher2, ZHER2 )
void BLAS_zher2(
    char const *uplo,
    blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *x, blas_int const *incx,
    blas_complex_double const *y, blas_int const *incy,
    blas_complex_double       *A, blas_int const *lda );

// -----------------------------------------------------------------------------
#define BLAS_strmv BLAS_FORTRAN_NAME( strmv, STRMV )
void BLAS_strmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    float const *A, blas_int const *lda,
    float       *x, blas_int const *incx );

#define BLAS_dtrmv BLAS_FORTRAN_NAME( dtrmv, DTRMV )
void BLAS_dtrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    double const *A, blas_int const *lda,
    double       *x, blas_int const *incx );

#define BLAS_ctrmv BLAS_FORTRAN_NAME( ctrmv, CTRMV )
void BLAS_ctrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float       *x, blas_int const *incx );

#define BLAS_ztrmv BLAS_FORTRAN_NAME( ztrmv, ZTRMV )
void BLAS_ztrmv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double       *x, blas_int const *incx );

// -----------------------------------------------------------------------------
#define BLAS_strsv BLAS_FORTRAN_NAME( strsv, STRSV )
void BLAS_strsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    float const *A, blas_int const *lda,
    float       *x, blas_int const *incx );

#define BLAS_dtrsv BLAS_FORTRAN_NAME( dtrsv, DTRSV )
void BLAS_dtrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    double const *A, blas_int const *lda,
    double       *x, blas_int const *incx );

#define BLAS_ctrsv BLAS_FORTRAN_NAME( ctrsv, CTRSV )
void BLAS_ctrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float       *x, blas_int const *incx );

#define BLAS_ztrsv BLAS_FORTRAN_NAME( ztrsv, ZTRSV )
void BLAS_ztrsv(
    char const *uplo, char const *trans, char const *diag,
    blas_int const *n,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double       *x, blas_int const *incx );

// =============================================================================
// Level 3 BLAS - Fortran prototypes

// -----------------------------------------------------------------------------
#define BLAS_sgemm BLAS_FORTRAN_NAME( sgemm, SGEMM )
void BLAS_sgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define BLAS_dgemm BLAS_FORTRAN_NAME( dgemm, DGEMM )
void BLAS_dgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define BLAS_cgemm BLAS_FORTRAN_NAME( cgemm, CGEMM )
void BLAS_cgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *B, blas_int const *ldb,
    blas_complex_float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zgemm BLAS_FORTRAN_NAME( zgemm, ZGEMM )
void BLAS_zgemm(
    char const *transA, char const *transB,
    blas_int const *m, blas_int const *n, blas_int const *k,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *B, blas_int const *ldb,
    blas_complex_double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define BLAS_ssymm BLAS_FORTRAN_NAME( ssymm, SSYMM )
void BLAS_ssymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define BLAS_dsymm BLAS_FORTRAN_NAME( dsymm, DSYMM )
void BLAS_dsymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define BLAS_csymm BLAS_FORTRAN_NAME( csymm, CSYMM )
void BLAS_csymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *B, blas_int const *ldb,
    blas_complex_float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zsymm BLAS_FORTRAN_NAME( zsymm, ZSYMM )
void BLAS_zsymm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *B, blas_int const *ldb,
    blas_complex_double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define BLAS_chemm BLAS_FORTRAN_NAME( chemm, CHEMM )
void BLAS_chemm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *B, blas_int const *ldb,
    blas_complex_float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zhemm BLAS_FORTRAN_NAME( zhemm, ZHEMM )
void BLAS_zhemm(
    char const *side, char const *uplo,
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *B, blas_int const *ldb,
    blas_complex_double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define BLAS_ssyrk BLAS_FORTRAN_NAME( ssyrk, SSYRK )
void BLAS_ssyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *beta,
    float       *C, blas_int const *ldc );

#define BLAS_dsyrk BLAS_FORTRAN_NAME( dsyrk, DSYRK )
void BLAS_dsyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *beta,
    double       *C, blas_int const *ldc );

#define BLAS_csyrk BLAS_FORTRAN_NAME( csyrk, CSYRK )
void BLAS_csyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zsyrk BLAS_FORTRAN_NAME( zsyrk, ZSYRK )
void BLAS_zsyrk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
// alpha and beta are real
#define BLAS_cherk BLAS_FORTRAN_NAME( cherk, CHERK )
void BLAS_cherk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zherk BLAS_FORTRAN_NAME( zherk, ZHERK )
void BLAS_zherk(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define BLAS_ssyr2k BLAS_FORTRAN_NAME( ssyr2k, SSYR2K )
void BLAS_ssyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    float const *alpha,
    float const *A, blas_int const *lda,
    float const *B, blas_int const *ldb,
    float const *beta,
    float       *C, blas_int const *ldc );

#define BLAS_dsyr2k BLAS_FORTRAN_NAME( dsyr2k, DSYR2K )
void BLAS_dsyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    double const *alpha,
    double const *A, blas_int const *lda,
    double const *B, blas_int const *ldb,
    double const *beta,
    double       *C, blas_int const *ldc );

#define BLAS_csyr2k BLAS_FORTRAN_NAME( csyr2k, CSYR2K )
void BLAS_csyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *B, blas_int const *ldb,
    blas_complex_float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zsyr2k BLAS_FORTRAN_NAME( zsyr2k, ZSYR2K )
void BLAS_zsyr2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *B, blas_int const *ldb,
    blas_complex_double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
// beta is real
#define BLAS_cher2k BLAS_FORTRAN_NAME( cher2k, CHER2K )
void BLAS_cher2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float const *B, blas_int const *ldb,
    float const *beta,
    blas_complex_float       *C, blas_int const *ldc );

#define BLAS_zher2k BLAS_FORTRAN_NAME( zher2k, ZHER2K )
void BLAS_zher2k(
    char const *uplo, char const *transA,
    blas_int const *n, blas_int const *k,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double const *B, blas_int const *ldb,
    double const *beta,
    blas_complex_double       *C, blas_int const *ldc );

// -----------------------------------------------------------------------------
#define BLAS_strmm BLAS_FORTRAN_NAME( strmm, STRMM )
void BLAS_strmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float       *B, blas_int const *ldb );

#define BLAS_dtrmm BLAS_FORTRAN_NAME( dtrmm, DTRMM )
void BLAS_dtrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double       *B, blas_int const *ldb );

#define BLAS_ctrmm BLAS_FORTRAN_NAME( ctrmm, CTRMM )
void BLAS_ctrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float       *B, blas_int const *ldb );

#define BLAS_ztrmm BLAS_FORTRAN_NAME( ztrmm, ZTRMM )
void BLAS_ztrmm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double       *B, blas_int const *ldb );

// -----------------------------------------------------------------------------
#define BLAS_strsm BLAS_FORTRAN_NAME( strsm, STRSM )
void BLAS_strsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    float const *alpha,
    float const *A, blas_int const *lda,
    float       *B, blas_int const *ldb );

#define BLAS_dtrsm BLAS_FORTRAN_NAME( dtrsm, DTRSM )
void BLAS_dtrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    double const *alpha,
    double const *A, blas_int const *lda,
    double       *B, blas_int const *ldb );

#define BLAS_ctrsm BLAS_FORTRAN_NAME( ctrsm, CTRSM )
void BLAS_ctrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    blas_complex_float const *alpha,
    blas_complex_float const *A, blas_int const *lda,
    blas_complex_float       *B, blas_int const *ldb );

#define BLAS_ztrsm BLAS_FORTRAN_NAME( ztrsm, ZTRSM )
void BLAS_ztrsm(
    char const *side, char const *uplo, char const *trans, char const *diag,
    blas_int const *m, blas_int const *n,
    blas_complex_double const *alpha,
    blas_complex_double const *A, blas_int const *lda,
    blas_complex_double       *B, blas_int const *ldb );

#ifdef __cplusplus
}  // #endif
#endif

#endif        //  #ifndef BLAS_FORTRAN_H
