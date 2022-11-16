// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_NAMES_HH
#define BLAS_DEVICE_NAMES_HH

#include "blas/device.hh"

#include <complex>

namespace blas {
namespace device {

// =============================================================================
// Level 1 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
// axpy
void saxpy(
    blas::Queue& queue,
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy);

void daxpy(
    blas::Queue& queue,
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy);

void caxpy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy);

void zaxpy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy);

// -----------------------------------------------------------------------------
// dot
void sdot(
    blas::Queue& queue,
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    float *result);

void ddot(
    blas::Queue& queue,
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    double *result);

void cdotu(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    std::complex<float> *result);

void zdotu(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    std::complex<double> *result);

void cdotc(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    std::complex<float> *result);

void zdotc(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    std::complex<double> *result);

// -----------------------------------------------------------------------------
// nrm2
void snrm2(
    blas::Queue& queue,
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *result);

void dnrm2(
    blas::Queue& queue,
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *result);

void cnrm2(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    float *result);

void znrm2(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    double *result);

// -----------------------------------------------------------------------------
// scal
void sscal(
    blas::Queue& queue,
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx);

void dscal(
    blas::Queue& queue,
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx);

void cscal(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx);

void zscal(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx);

// -----------------------------------------------------------------------------
// swap
void sswap(
    blas::Queue& queue,
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy);

void dswap(
    blas::Queue& queue,
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy);

void cswap(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy);

void zswap(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy);

// -----------------------------------------------------------------------------
void scopy(
    blas::Queue& queue,
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy);

void dcopy(
    blas::Queue& queue,
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy);

void ccopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy);

void zcopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy);


// =============================================================================
// Level 2 BLAS - Device Interfaces

// -----------------------------------------------------------------------------

// =============================================================================
// Level 3 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
// gemm
void sgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc);

void dgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc);

void cgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc);

void zgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// trsm
void strsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb);

void dtrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb);

void ctrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb);

void ztrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb);

// -----------------------------------------------------------------------------
// trmm
void strmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb);

void dtrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb);

void ctrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb);

void ztrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb);

// -----------------------------------------------------------------------------
// hemm
void chemm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zhemm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// symm
void ssymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc);

void dsymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc);

void csymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// herk
void cherk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zherk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// syrk
void ssyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc);

void dsyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc);

void csyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// her2k
void cher2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zher2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// syr2k
void ssyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc);

void dsyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc);

void csyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// batch gemm
void batch_sgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_dgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_cgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_zgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

// -----------------------------------------------------------------------------
// batch trsm
void batch_strsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_dtrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_ctrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_ztrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

}  // namespace device
}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_NAMES_HH
