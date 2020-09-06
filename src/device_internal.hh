// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_NAMES_HH
#define BLAS_DEVICE_NAMES_HH

#include "blas/device_types.hh"

#include <complex>

// constants -- defined as needed.
// Only needed are those shared between cublas and rocblas
// Constants that exist on only one side should be explicitly handled
#define    DevSuccess        cudaSuccess
#define    DevBlasSuccess    CUBLAS_STATUS_SUCCESS

// trans
#define    DevNoTrans        CUBLAS_OP_N
#define    DevTrans          CUBLAS_OP_T
#define    DevConjTrans      CUBLAS_OP_C

// diag
#define    DevDiagUnit       CUBLAS_DIAG_UNIT
#define    DevDiagNonUnit    CUBLAS_DIAG_NON_UNIT

// uplo
#define    DevUploUpper      CUBLAS_FILL_MODE_UPPER
#define    DevUploLower      CUBLAS_FILL_MODE_LOWER

// side
#define    DevSideLeft       CUBLAS_SIDE_LEFT
#define    DevSideRight      CUBLAS_SIDE_RIGHT

namespace blas {
namespace device {

// =============================================================================
// Level 1 BLAS - Device Interfaces

// -----------------------------------------------------------------------------

// =============================================================================
// Level 2 BLAS - Device Interfaces

// -----------------------------------------------------------------------------

// =============================================================================
// Level 3 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
// gemm
void sgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc);

void dgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc);

void cgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc);

void zgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// trsm
void strsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb);

void dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb);

void ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb);

void ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb);

// -----------------------------------------------------------------------------
// trmm
void strmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb);

void dtrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb);

void ctrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb);

void ztrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb);

// -----------------------------------------------------------------------------
// hemm
void chemm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zhemm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// symm
void ssymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc);

void dsymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc);

void csymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// herk
void cherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// syrk
void ssyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc);

void dsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc);

void csyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// her2k
void cher2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zher2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// syr2k
void ssyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc);

void dsyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc);

void csyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc);

void zsyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc);

// -----------------------------------------------------------------------------
// batch gemm
void batch_sgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_dgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_cgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size);

void batch_zgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
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
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

void batch_ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size);

}  // namespace device
}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_NAMES_HH
