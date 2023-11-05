// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

// Moved from lapack_wrappers.hh for ESSL compatability

// get BLAS_FORTRAN_NAME and blas_int
#include "blas/fortran.h"

#include "lapack_wrappers.hh"

#include <complex>

// This is a temporary file giving simple LAPACK wrappers,
// until the real lapackpp wrappers are available.

// -----------------------------------------------------------------------------
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
                     int64_t m, int64_t n,
                     float const *A, int64_t lda,
                     float *work )
{
    blas_int lda_ = (blas_int) lda;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    return lapack_slange( norm, &m_, &n_, A, &lda_, work );
}

double lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     double const *A, int64_t lda,
                     double *work )
{
    blas_int lda_ = (blas_int) lda;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    return lapack_dlange( norm, &m_, &n_, A, &lda_, work );
}

float  lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work )
{
    blas_int lda_ = (blas_int) lda;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    return lapack_clange( norm, &m_, &n_, A, &lda_, work );
}

double lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work )
{
    blas_int lda_ = (blas_int) lda;
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    return lapack_zlange( norm, &m_, &n_, A, &lda_, work );
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
                     int64_t n,
                     float const *A, int64_t lda,
                     float *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_slansy( norm, uplo, &n_, A, &lda_, work );
}

double lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     double const *A, int64_t lda,
                     double *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_dlansy( norm, uplo, &n_, A, &lda_, work );
}

float  lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_clansy( norm, uplo, &n_, A, &lda_, work );
}

double lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_zlansy( norm, uplo, &n_, A, &lda_, work );
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
                     int64_t n,
                     float const *A, int64_t lda,
                     float *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_slansy( norm, uplo, &n_, A, &lda_, work );
}

double lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     double const *A, int64_t lda,
                     double *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_dlansy( norm, uplo, &n_, A, &lda_, work );
}

float  lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_clanhe( norm, uplo, &n_, A, &lda_, work );
}

double lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_zlanhe( norm, uplo, &n_, A, &lda_, work );
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
                     int64_t m, int64_t n,
                     float const *A, int64_t lda,
                     float *work )
{
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_slantr( norm, uplo, diag, &m_, &n_, A, &lda_, work );
}

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     double const *A, int64_t lda,
                     double *work )
{
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_dlantr( norm, uplo, diag, &m_, &n_, A, &lda_, work );
}

float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work )
{
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_clantr( norm, uplo, diag, &m_, &n_, A, &lda_, work );
}

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work )
{
    blas_int m_ = (blas_int) m;
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    return lapack_zlantr( norm, uplo, diag, &m_, &n_, A, &lda_, work );
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
void lapack_potrf(  char const *uplo, int64_t n,
                    float *A, int64_t lda,
                    int64_t *info )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int *info_ = (blas_int *) info;
    lapack_spotrf( uplo, &n_, A, &lda_, info_ );
}

void lapack_potrf(  char const *uplo, int64_t n,
                    double *A, int64_t lda,
                    int64_t *info )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int *info_ = (blas_int *) info;
    lapack_dpotrf( uplo, &n_, A, &lda_, info_ );
}

void lapack_potrf(  char const *uplo, int64_t n,
                    std::complex<float> *A, int64_t lda,
                    int64_t *info )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int *info_ = (blas_int *) info;
    lapack_cpotrf( uplo, &n_, A, &lda_, info_ );
}

void lapack_potrf(  char const *uplo, int64_t n,
                    std::complex<double> *A, int64_t lda,
                    int64_t *info )
{
    blas_int n_ = (blas_int) n;
    blas_int lda_ = (blas_int) lda;
    blas_int *info_ = (blas_int *) info;
    lapack_zpotrf( uplo, &n_, A, &lda_, info_ );
}
