// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <vector>

#include "blas/util.hh"

namespace blas {

// =============================================================================
// Level 1 BLAS

// -----------------------------------------------------------------------------
/// @ingroup asum
float asum(
    int64_t n,
    float const *x, int64_t incx );

/// @ingroup asum
double asum(
    int64_t n,
    double const *x, int64_t incx );

/// @ingroup asum
float asum(
    int64_t n,
    std::complex<float> const *x, int64_t incx );

/// @ingroup asum
double asum(
    int64_t n,
    std::complex<double> const *x, int64_t incx );

// -----------------------------------------------------------------------------
/// @ingroup axpy
void axpy(
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *y, int64_t incy );

/// @ingroup axpy
void axpy(
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *y, int64_t incy );

/// @ingroup axpy
void axpy(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *y, int64_t incy );

/// @ingroup axpy
void axpy(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup copy
void copy(
    int64_t n,
    float const *x, int64_t incx,
    float       *y, int64_t incy );

/// @ingroup copy
void copy(
    int64_t n,
    double const *x, int64_t incx,
    double       *y, int64_t incy );

/// @ingroup copy
void copy(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *y, int64_t incy );

/// @ingroup copy
void copy(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup dot
float dot(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy );

/// @ingroup dot
double dot(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy );

/// @ingroup dot
std::complex<float> dot(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy );

/// @ingroup dot
std::complex<double> dot(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup dotu
float dotu(
    int64_t n,
    float const *x, int64_t incx,
    float const *y, int64_t incy );

/// @ingroup dotu
double dotu(
    int64_t n,
    double const *x, int64_t incx,
    double const *y, int64_t incy );

/// @ingroup dotu
std::complex<float> dotu(
    int64_t n,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy );

/// @ingroup dotu
std::complex<double> dotu(
    int64_t n,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup iamax
int64_t iamax(
    int64_t n,
    float const *x, int64_t incx );

/// @ingroup iamax
int64_t iamax(
    int64_t n,
    double const *x, int64_t incx );

/// @ingroup iamax
int64_t iamax(
    int64_t n,
    std::complex<float> const *x, int64_t incx );

/// @ingroup iamax
int64_t iamax(
    int64_t n,
    std::complex<double> const *x, int64_t incx );

// -----------------------------------------------------------------------------
/// @ingroup nrm2
float nrm2(
    int64_t n,
    float const *x, int64_t incx );

/// @ingroup nrm2
double nrm2(
    int64_t n,
    double const *x, int64_t incx );

/// @ingroup nrm2
float nrm2(
    int64_t n,
    std::complex<float> const *x, int64_t incx );

/// @ingroup nrm2
double nrm2(
    int64_t n,
    std::complex<double> const *x, int64_t incx );

// -----------------------------------------------------------------------------
/// @ingroup rot
void rot(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy,
    float c,
    float s );

/// @ingroup rot
void rot(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy,
    double c,
    double s );

/// @ingroup rot
// real cosine, real sine
void rot(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy,
    float c,
    float s );

/// @ingroup rot
// real cosine, real sine
void rot(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy,
    double c,
    double s );

/// @ingroup rot
// real cosine, complex sine
void rot(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy,
    float c,
    std::complex<float> s );

/// @ingroup rot
// real cosine, complex sine
void rot(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy,
    double c,
    std::complex<double> s );

// -----------------------------------------------------------------------------
/// @ingroup rotg
void rotg(
    float *a,
    float *b,
    float *c,
    float *s );

/// @ingroup rotg
void rotg(
    double *a,
    double *b,
    double *c,
    double *s );

/// @ingroup rotg
void rotg(
    std::complex<float> *a,
    std::complex<float> *b,  // const in BLAS implementation, oddly
    float *c,
    std::complex<float> *s );

/// @ingroup rotg
void rotg(
    std::complex<double> *a,
    std::complex<double> *b,  // const in BLAS implementation, oddly
    double *c,
    std::complex<double> *s );

// -----------------------------------------------------------------------------
// only real
/// @ingroup rotm
void rotm(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy,
    float const param[5] );

/// @ingroup rotm
void rotm(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy,
    double const param[5] );

// -----------------------------------------------------------------------------
// only real
/// @ingroup rotmg
void rotmg(
    float *d1,
    float *d2,
    float *a,
    float  b,
    float  param[5] );

/// @ingroup rotmg
void rotmg(
    double *d1,
    double *d2,
    double *a,
    double  b,
    double  param[5] );

// -----------------------------------------------------------------------------
/// @ingroup scal
void scal(
    int64_t n,
    float alpha,
    float *x, int64_t incx );

/// @ingroup scal
void scal(
    int64_t n,
    double alpha,
    double *x, int64_t incx );

/// @ingroup scal
void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> *x, int64_t incx );

/// @ingroup scal
void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> *x, int64_t incx );

// -----------------------------------------------------------------------------
/// @ingroup swap
void swap(
    int64_t n,
    float *x, int64_t incx,
    float *y, int64_t incy );

/// @ingroup swap
void swap(
    int64_t n,
    double *x, int64_t incx,
    double *y, int64_t incy );

/// @ingroup swap
void swap(
    int64_t n,
    std::complex<float> *x, int64_t incx,
    std::complex<float> *y, int64_t incy );

/// @ingroup swap
void swap(
    int64_t n,
    std::complex<double> *x, int64_t incx,
    std::complex<double> *y, int64_t incy );
// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy );

/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy );

/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy );

/// @ingroup gemv
void gemv(
    blas::Layout layout,
    blas::Op trans,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda );

/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda );

/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda );

/// @ingroup ger
void ger(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda );

/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda );

/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda );

/// @ingroup geru
void geru(
    blas::Layout layout,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy );

/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy );

/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy );

/// @ingroup hemv
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy );

// -----------------------------------------------------------------------------
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *A, int64_t lda );

/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *A, int64_t lda );

/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *A, int64_t lda );

/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda );

/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda );

/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda );

/// @ingroup her2
void her2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *x, int64_t incx,
    float beta,
    float       *y, int64_t incy );

/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *x, int64_t incx,
    double beta,
    double       *y, int64_t incy );

/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>       *y, int64_t incy );

/// @ingroup symv
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>       *y, int64_t incy );

// -----------------------------------------------------------------------------
// only real; complex in lapack++
/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *A, int64_t lda );

/// @ingroup syr
void syr(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda );

/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda );

/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
    std::complex<float>       *A, int64_t lda );

/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
    std::complex<double>       *A, int64_t lda );

// -----------------------------------------------------------------------------
/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const *A, int64_t lda,
    float       *x, int64_t incx );

/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const *A, int64_t lda,
    double       *x, int64_t incx );

/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *x, int64_t incx );

/// @ingroup trmv
void trmv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *x, int64_t incx );

// -----------------------------------------------------------------------------
/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const *A, int64_t lda,
    float       *x, int64_t incx );

/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const *A, int64_t lda,
    double       *x, int64_t incx );

/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *x, int64_t incx );

/// @ingroup trsv
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *x, int64_t incx );
// =============================================================================
// Level 3 BLAS

// -----------------------------------------------------------------------------
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc );

/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc );

/// @ingroup hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup her2k
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup her2k
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup her2k
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,  // note: complex
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    float beta,   // note: real
    std::complex<float>       *C, int64_t ldc );

/// @ingroup her2k
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,  // note: complex
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    double beta,  // note: real
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,  // note: real
    std::complex<float> const *A, int64_t lda,
    float beta,   // note: real
    std::complex<float>       *C, int64_t ldc );

/// @ingroup herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    std::complex<double> const *A, int64_t lda,
    double beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup symm
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup symm
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup symm
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc );

/// @ingroup symm
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float const *B, int64_t ldb,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double const *B, int64_t ldb,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> const *B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc );

/// @ingroup syr2k
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> const *B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *A, int64_t lda,
    float beta,
    float       *C, int64_t ldc );

/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *A, int64_t lda,
    double beta,
    double       *C, int64_t ldc );

/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float> beta,
    std::complex<float>       *C, int64_t ldc );

/// @ingroup syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double> beta,
    std::complex<double>       *C, int64_t ldc );

// -----------------------------------------------------------------------------
/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float       *B, int64_t ldb );

/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double       *B, int64_t ldb );

/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *B, int64_t ldb );

/// @ingroup trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *B, int64_t ldb );

// -----------------------------------------------------------------------------
/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *A, int64_t lda,
    float       *B, int64_t ldb );

/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *A, int64_t lda,
    double       *B, int64_t ldb );

/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *B, int64_t ldb );

/// @ingroup trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *B, int64_t ldb );

// =============================================================================
//                     Batch BLAS APIs ( host )
// =============================================================================
namespace batch {

// -----------------------------------------------------------------------------
// batch gemm
void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<float >   const &alpha,
    std::vector<float*>   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>   const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >   const &beta,
    std::vector<float*>   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<double >  const &alpha,
    std::vector<double*>  const &Aarray, std::vector<int64_t>  const &ldda,
    std::vector<double*>  const &Barray, std::vector<int64_t>  const &lddb,
    std::vector<double >  const &beta,
    std::vector<double*>  const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<float>  >   const &alpha,
    std::vector< std::complex<float>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<float>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<float>  >   const &beta,
    std::vector< std::complex<float>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                  std::vector<int64_t>  &info );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<double>  >   const &alpha,
    std::vector< std::complex<double>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<double>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<double>  >   const &beta,
    std::vector< std::complex<double>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                   std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch trsm
void trsm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                    std::vector<int64_t>       &info );

void trsm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

void trsm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

void trsm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch trmm
void trmm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                    std::vector<int64_t>       &info );

void trmm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

void trmm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

void trmm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch hemm
void hemm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void hemm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void hemm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void hemm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch symm
void symm(
    blas::Layout                   layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void symm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void symm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void symm(
    blas::Layout                    layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch herk
void herk(
    blas::Layout                   layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void herk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info );

void herk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<float>       const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >      const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

void herk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double>      const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch syrk
void syrk(
    blas::Layout                   layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void syrk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info );

void syrk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float> > const &alpha,
    std::vector<std::complex<float>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float> > const &beta,
    std::vector<std::complex<float>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

void syrk(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double> > const &alpha,
    std::vector<std::complex<double>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double> > const &beta,
    std::vector<std::complex<double>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch her2k
void her2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void her2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info );

void her2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float>>      const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >                   const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

void her2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double>>      const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >                   const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

// -----------------------------------------------------------------------------
// batch syr2k
void syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info );

void syr2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info );

void syr2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float> > const &alpha,
    std::vector<std::complex<float>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> > const &beta,
    std::vector<std::complex<float>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

void syr2k(
    blas::Layout                    layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double> > const &alpha,
    std::vector<std::complex<double>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> > const &beta,
    std::vector<std::complex<double>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info );

}  // namespace batch
}  // namespace blas
