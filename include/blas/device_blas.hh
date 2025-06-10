// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#include <vector>

namespace blas {

//==============================================================================
// Level 1 BLAS
// Alphabetical order

//------------------------------------------------------------------------------
void asum(
    int64_t n,
    float const* x, int64_t incx,
    float* result,
    blas::Queue& queue );

void asum(
    int64_t n,
    double const* x, int64_t incx,
    double* result,
    blas::Queue& queue );

void asum(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    float* result,
    blas::Queue& queue );

void asum(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    double* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void axpy(
    int64_t n,
    float alpha,
    float const* x, int64_t incx,
    float*       y, int64_t incy,
    blas::Queue& queue );

void axpy(
    int64_t n,
    double alpha,
    double const* x, int64_t incx,
    double*       y, int64_t incy,
    blas::Queue& queue );

void axpy(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>*       y, int64_t incy,
    blas::Queue& queue );

void axpy(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>*       y, int64_t incy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void copy(
    int64_t n,
    float const* x, int64_t incx,
    float*       y, int64_t incy,
    blas::Queue& queue );

void copy(
    int64_t n,
    double const* x, int64_t incx,
    double*       y, int64_t incy,
    blas::Queue& queue );

void copy(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float>*       y, int64_t incy,
    blas::Queue& queue );

void copy(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double>*       y, int64_t incy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void dot(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float* result,
    blas::Queue& queue );

void dot(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double* result,
    blas::Queue& queue );

void dot(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>* result,
    blas::Queue& queue );

void dot(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void dotu(
    int64_t n,
    float const* x, int64_t incx,
    float const* y, int64_t incy,
    float* result,
    blas::Queue& queue );

void dotu(
    int64_t n,
    double const* x, int64_t incx,
    double const* y, int64_t incy,
    double* result,
    blas::Queue& queue );

void dotu(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> const* y, int64_t incy,
    std::complex<float>* result,
    blas::Queue& queue );

void dotu(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> const* y, int64_t incy,
    std::complex<double>* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    float const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    double const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

void iamax(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    int64_t* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void nrm2(
    int64_t n,
    float const* x, int64_t incx,
    float* result,
    blas::Queue& queue );

void nrm2(
    int64_t n,
    double const* x, int64_t incx,
    double* result,
    blas::Queue& queue );

void nrm2(
    int64_t n,
    std::complex<float> const* x, int64_t incx,
    float* result,
    blas::Queue& queue );

void nrm2(
    int64_t n,
    std::complex<double> const* x, int64_t incx,
    double* result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rot(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy,
    const float c,
    const float s,
    blas::Queue& queue );

void rot(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy,
    const double c,
    const double s,
    blas::Queue& queue );

void rot(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    const float c,
    const float s,
    blas::Queue& queue );

void rot(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    const double c,
    const double s,
    blas::Queue& queue );

void rot(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    const float c,
    const std::complex<float> s,
    blas::Queue& queue );

void rot(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    const double c,
    const std::complex<double> s,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotg(
    float* a,
    float* b,
    float* c,
    float* s,
    blas::Queue& queue );

void rotg(
    double* a,
    double* b,
    double* c,
    double* s,
    blas::Queue& queue );

void rotg(
    std::complex<float>* a,
    std::complex<float>* b,
    float* c,
    std::complex<float>* s,
    blas::Queue& queue );

void rotg(
    std::complex<double>* a,
    std::complex<double>* b,
    double* c,
    std::complex<double>* s,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotm(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy,
    const float* param,
    blas::Queue& queue );

void rotm(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy,
    const double* param,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void rotmg(
    float* d1,
    float* d2,
    float* x1,
    float* y1,
    float* param,
    blas::Queue& queue );

void rotmg(
    double* d1,
    double* d2,
    double* x1,
    double* y1,
    double* param,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void scal(
    int64_t n,
    float alpha,
    float* x, int64_t incx,
    blas::Queue& queue );

void scal(
    int64_t n,
    double alpha,
    double* x, int64_t incx,
    blas::Queue& queue );

void scal(
    int64_t n,
    std::complex<float> alpha,
    std::complex<float>* x, int64_t incx,
    blas::Queue& queue );

void scal(
    int64_t n,
    std::complex<double> alpha,
    std::complex<double>* x, int64_t incx,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void swap(
    int64_t n,
    float* x, int64_t incx,
    float* y, int64_t incy,
    blas::Queue& queue );

void swap(
    int64_t n,
    double* x, int64_t incx,
    double* y, int64_t incy,
    blas::Queue& queue );

void swap(
    int64_t n,
    std::complex<float>* x, int64_t incx,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue );

void swap(
    int64_t n,
    std::complex<double>* x, int64_t incx,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue );

//==============================================================================
// Level 2 BLAS

//------------------------------------------------------------------------------
void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float* y, int64_t incy,
    blas::Queue& queue );

void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double* y, int64_t incy,
    blas::Queue& queue );

void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue );

void hemv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* x, int64_t incx,
    float beta,
    float* y, int64_t incy,
    blas::Queue& queue );

void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* x, int64_t incx,
    double beta,
    double* y, int64_t incy,
    blas::Queue& queue );

void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* x, int64_t incx,
    std::complex<float> beta,
    std::complex<float>* y, int64_t incy,
    blas::Queue& queue );

void symv(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* x, int64_t incx,
    std::complex<double> beta,
    std::complex<double>* y, int64_t incy,
    blas::Queue& queue );

//==============================================================================
// Level 3 BLAS

//------------------------------------------------------------------------------
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,  // note: complex
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    float beta,   // note: real
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,  // note: complex
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    double beta,  // note: real
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,  // note: real
    std::complex<float> const* A, int64_t lda,
    float beta,   // note: real
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    std::complex<double> const* A, int64_t lda,
    double beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float beta,
    float*       C, int64_t ldc,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double beta,
    double*       C, int64_t ldc,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float*       B, int64_t ldb,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double*       B, int64_t ldb,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       B, int64_t ldb,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       B, int64_t ldb,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const* A, int64_t lda,
    float*       B, int64_t ldb,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const* A, int64_t lda,
    double*       B, int64_t ldb,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float>*       B, int64_t ldb,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double>*       B, int64_t ldb,
    blas::Queue& queue );

//==============================================================================
//                     Batch BLAS APIs (device)
//==============================================================================
namespace batch {

//==============================================================================
// Level 1 Batch BLAS

//==============================================================================
// Level 2 Batch BLAS

//==============================================================================
// Level 3 Batch BLAS

//------------------------------------------------------------------------------
// batch gemm
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch gemm, group API
void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void gemm(
    blas::Layout layout,
    std::vector<blas::Op>   const& transA,
    std::vector<blas::Op>   const& transB,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    std::vector<size_t>     const& group_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch hemm
void hemm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void hemm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch her2k
void her2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< float >                const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void her2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< double >                const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch herk
void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< float >                const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< float >                const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void herk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< double >                const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< double >                const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch symm
void symm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void symm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch syr2k
void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syr2k(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch syrk
void syrk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float >     const& beta,
    std::vector<float*>     const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double >    const& beta,
    std::vector<double*>    const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>  > const& beta,
    std::vector< std::complex<float>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void syrk(
    blas::Layout layout,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<int64_t>    const& n,
    std::vector<int64_t>    const& k,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>  > const& beta,
    std::vector< std::complex<double>* > const& Carray, std::vector<int64_t> const& ldc,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch trmm
void trmm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trmm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch trsm
void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<float >     const& alpha,
    std::vector<float*>     const& Aarray, std::vector<int64_t> const& lda,
    std::vector<float*>     const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector<double >    const& alpha,
    std::vector<double*>    const& Aarray, std::vector<int64_t> const& lda,
    std::vector<double*>    const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<float>  > const& alpha,
    std::vector< std::complex<float>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<float>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

void trsm(
    blas::Layout layout,
    std::vector<blas::Side> const& side,
    std::vector<blas::Uplo> const& uplo,
    std::vector<blas::Op>   const& trans,
    std::vector<blas::Diag> const& diag,
    std::vector<int64_t>    const& m,
    std::vector<int64_t>    const& n,
    std::vector< std::complex<double>  > const& alpha,
    std::vector< std::complex<double>* > const& Aarray, std::vector<int64_t> const& lda,
    std::vector< std::complex<double>* > const& Barray, std::vector<int64_t> const& ldb,
    size_t batch_size,
    std::vector<int64_t>& info,
    blas::Queue& queue );

}  // namespace batch
}  // namespace blas
