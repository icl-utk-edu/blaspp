// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#ifdef BLAS_HAVE_ROCBLAS

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
// -----------------------------------------------------------------------------
// sgemm
void sgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_sgemm(
            handle, transA, transB,
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dgemm
void dgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_dgemm(
            handle, transA, transB,
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// cgemm
void cgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_cgemm(
            handle, transA, transB, m, n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zgemm
void zgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zgemm(
            handle, transA, transB,
            m, n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// trsm
// -----------------------------------------------------------------------------
// strsm
void strsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_strsm(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// dtrsm
void dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_dtrsm(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ctrsm
void ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_ctrsm(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ztrsm
void ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_ztrsm(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb ) );
}

// -----------------------------------------------------------------------------
// trmm
// -----------------------------------------------------------------------------
// strmm
void strmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_strmm(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// dtrmm
void dtrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_dtrmm(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ctrmm
void ctrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_ctrmm(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ztrmm
void ztrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    blas_dev_call(
        rocblas_ztrmm(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb ) );
}

// -----------------------------------------------------------------------------
// hemm
// -----------------------------------------------------------------------------
// chemm
void chemm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_chemm(
            handle, side, uplo,
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zhemm
void zhemm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zhemm(
            handle, side, uplo,
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// symm
// -----------------------------------------------------------------------------
// ssymm
void ssymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_ssymm(
            handle, side, uplo,
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsymm
void dsymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_dsymm(
            handle, side, uplo,
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csymm
void csymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_csymm(
            handle, side, uplo,
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsymm
void zsymm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zsymm(
            handle, side, uplo,
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// herk
// -----------------------------------------------------------------------------
// cherk
void cherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_cherk(
            handle, uplo, trans,
            n, k,
            &alpha, (rocblas_float_complex*) dA, ldda,
            &beta,  (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zherk
void zherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zherk(
            handle, uplo, trans,
            n, k,
            &alpha, (rocblas_double_complex*) dA, ldda,
            &beta,  (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// syrk
// -----------------------------------------------------------------------------
// ssyrk
void ssyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_ssyrk(
            handle, uplo, trans,
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsyrk
void dsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_dsyrk(
            handle, uplo, trans,
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csyrk
void csyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_csyrk(
            handle, uplo, trans,
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsyrk
void zsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zsyrk(
            handle, uplo, trans,
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// her2k
// -----------------------------------------------------------------------------
// cher2k
void cher2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_cher2k(
            handle, uplo, trans,
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zher2k
void zher2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zher2k(
            handle, uplo, trans,
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// syr2k
// -----------------------------------------------------------------------------
// ssyr2k
void ssyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_ssyr2k(
            handle, uplo, trans,
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsyr2k
void dsyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_dsyr2k(
            handle, uplo, trans,
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csyr2k
void csyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_csyr2k(
            handle, uplo, trans,
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsyr2k
void zsyr2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    blas_dev_call(
        rocblas_zsyr2k(
            handle, uplo, trans,
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
}

// -----------------------------------------------------------------------------
// batch gemm
// -----------------------------------------------------------------------------
// batch sgemm
void batch_sgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_sgemm_batched(
            handle, transA, transB,
            m, n, k,
            &alpha,
            (float const**) dAarray, ldda,
            (float const**) dBarray, lddb,
            &beta,
            dCarray, lddc,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch dgemm
void batch_dgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_dgemm_batched(
            handle, transA, transB,
            m, n, k,
            &alpha,
            (double const**) dAarray, ldda,
            (double const**) dBarray, lddb,
            &beta,
            dCarray, lddc,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch cgemm
void batch_cgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_cgemm_batched(
            handle, transA, transB,
            m, n, k,
            (rocblas_float_complex*)        &alpha,
            (rocblas_float_complex const**) dAarray, ldda,
            (rocblas_float_complex const**) dBarray, lddb,
            (rocblas_float_complex*)        &beta,
            (rocblas_float_complex**)       dCarray, lddc,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch zgemm
void batch_zgemm(
    device_blas_handle_t handle,
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_zgemm_batched(
            handle, transA, transB,
            m, n, k,
            (rocblas_double_complex*)        &alpha,
            (rocblas_double_complex const**) dAarray, ldda,
            (rocblas_double_complex const**) dBarray, lddb,
            (rocblas_double_complex*)        &beta,
            (rocblas_double_complex**)       dCarray, lddc,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch strsm
void batch_strsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)

{
    blas_dev_call(
        rocblas_strsm_batched(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            (float const**) dAarray, ldda,
            (float**)       dBarray, lddb,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch dtrsm
void batch_dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_dtrsm_batched(
            handle, side, uplo, trans, diag,
            m, n,
            &alpha,
            (double const**) dAarray, ldda,
            (double**)       dBarray, lddb,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch ctrsm
void batch_ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_ctrsm_batched(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_float_complex*)        &alpha,
            (rocblas_float_complex const**) dAarray, ldda,
            (rocblas_float_complex**)       dBarray, lddb,
            batch_size ) );
}

// -----------------------------------------------------------------------------
// batch ztrsm
void batch_ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    blas_dev_call(
        rocblas_ztrsm_batched(
            handle, side, uplo, trans, diag,
            m, n,
            (rocblas_double_complex*)        &alpha,
            (rocblas_double_complex const**) dAarray, ldda,
            (rocblas_double_complex**)       dBarray, lddb,
            batch_size ) );
}

}  // namespace device
}  // namespace blas

#endif  // BLAS_HAVE_ROCBLAS
