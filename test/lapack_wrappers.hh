// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef LAPACK_WRAPPERS_HH
#define LAPACK_WRAPPERS_HH

// get BLAS_FORTRAN_NAME and int64_t
#include <cassert>
#include <complex>

// This is a temporary file giving simple LAPACK wrappers,
// until the real lapackpp wrappers are available.

// -----------------------------------------------------------------------------
template <typename TX>
void lapack_larnv( int64_t idist, int iseed[4], int64_t size, TX *x )
{
    for (int64_t i = 0; i < size; ++i) {
        x[i] = rand() / TX ( RAND_MAX );
    }
}

template <typename TX>
void lapack_larnv( int64_t idist, int iseed[4], int64_t size, std::complex <TX> *x )
{
    for (int64_t i = 0; i < size; ++i) {
        x[i] = std::complex <TX>( rand() / TX ( RAND_MAX ), rand() / TX ( RAND_MAX ));
    }
}

// -----------------------------------------------------------------------------
float  lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     float const *A, int64_t lda,
                     float *work );

double lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     double const *A, int64_t lda,
                     double *work );

float  lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work );

double lapack_lange( char const *norm,
                     int64_t m, int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     float const *A, int64_t lda,
                     float *work );

double lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     double const *A, int64_t lda,
                     double *work );

float  lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work );

double lapack_lansy( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     float const *A, int64_t lda,
                     float *work );

double lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     double const *A, int64_t lda,
                     double *work );

float  lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work );

double lapack_lanhe( char const *norm, char const *uplo,
                     int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work );

// -----------------------------------------------------------------------------
float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     float const *A, int64_t lda,
                     float *work );

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     double const *A, int64_t lda,
                     double *work );

float  lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     std::complex<float> const *A, int64_t lda,
                     float *work );

double lapack_lantr( char const *norm, char const *uplo, char const *diag,
                     int64_t m, int64_t n,
                     std::complex<double> const *A, int64_t lda,
                     double *work );

// -----------------------------------------------------------------------------
template <typename TA, typename TB>
void lapack_lacpy( char const* uplo,
                     int64_t m, int64_t n,
                     TA const *A, int64_t lda,
                     TB       *B, int64_t ldb )
{
    assert( tolower(*uplo) == 'g' );
    for (int64_t j = 0; j < n; ++j) {
        for (int64_t i = 0; i < m; ++i) {
            B[i + j * ldb] = A[i + j * lda];
        }
    }
}

// -----------------------------------------------------------------------------
void lapack_potrf(  char const *uplo, int64_t n,
                    float *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char const *uplo, int64_t n,
                    double *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char const *uplo, int64_t n,
                    std::complex<float> *A, int64_t lda,
                    int64_t *info );

void lapack_potrf(  char const *uplo, int64_t n,
                    std::complex<double> *A, int64_t lda,
                    int64_t *info );

#endif        //  #ifndef LAPACK_WRAPPERS_HH
