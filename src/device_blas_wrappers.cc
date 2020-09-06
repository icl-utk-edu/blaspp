// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSgemm(
                handle, transA, transB,
                m, n, k,
                &alpha, dA, ldda, dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDgemm(
                handle, transA, transB,
                m, n, k,
                &alpha, dA, ldda, dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCgemm(
                handle, transA, transB, m, n, k,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda, (cuComplex*)dB, lddb,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZgemm(
                handle, transA, transB,
                m, n, k,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda, (cuDoubleComplex*)dB, lddb,
                (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasStrsm(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                dA, ldda,
                dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDtrsm(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                dA, ldda,
                dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCtrsm(
                handle, side, uplo, trans, diag,
                m, n, (cuComplex*)&alpha,
                (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZtrsm(
                handle, side, uplo, trans, diag,
                m, n, (cuDoubleComplex*)&alpha,
                (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasStrmm(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                dA, ldda,
                dB, lddb,
                dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDtrmm(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                dA, ldda,
                dB, lddb,
                dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCtrmm(
                handle, side, uplo, trans, diag,
                m, n, (cuComplex*)&alpha,
                (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb,
                (cuComplex*)dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZtrmm(
                handle, side, uplo, trans, diag,
                m, n, (cuDoubleComplex*)&alpha,
                (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb,
                (cuDoubleComplex*)dB, lddb ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasChemm(
                handle, side, uplo,
                m, n,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZhemm(
                handle, side, uplo,
                m, n,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb,
                (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSsymm(
                handle, side, uplo,
                m, n,
                &alpha, dA, ldda,
                dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDsymm(
                handle, side, uplo,
                m, n,
                &alpha, dA, ldda,
                dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCsymm(
                handle, side, uplo,
                m, n,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZsymm(
                handle, side, uplo,
                m, n,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb,
                (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCherk(
                handle, uplo, trans,
                n, k,
                &alpha, (cuComplex*)dA, ldda,
                &beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZherk(
                handle, uplo, trans,
                n, k,
                &alpha, (cuDoubleComplex*)dA, ldda,
                &beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSsyrk(
                handle, uplo, trans,
                n, k,
                &alpha, dA, ldda,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDsyrk(
                handle, uplo, trans,
                n, k,
                &alpha, dA, ldda,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCsyrk(
                handle, uplo, trans,
                n, k,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZsyrk(
                handle, uplo, trans,
                n, k,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCher2k(
                handle, uplo, trans,
                n, k,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb,
                &beta,              (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// zher2k
void zher2k(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZher2k(
                handle, uplo, trans,
                n, k,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb,
                &beta,                    (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSsyr2k(
                handle, uplo, trans,
                n, k,
                &alpha, dA, ldda,
                dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDsyr2k(
                handle, uplo, trans,
                n, k,
                &alpha, dA, ldda,
                dB, lddb,
                &beta,  dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCsyr2k(
                handle, uplo, trans,
                n, k,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                (cuComplex*)dB, lddb,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZsyr2k(
                handle, uplo, trans,
                n, k,
                (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                (cuDoubleComplex*)dB, lddb,
                (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasSgemmBatched(
                handle, transA, transB,
                m, n, k,
                &alpha, (const float**)dAarray, ldda, (const float**)dBarray, lddb,
                &beta,                 dCarray, lddc, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDgemmBatched(
                handle, transA, transB,
                m, n, k,
                &alpha, (const double**)dAarray, ldda, (const double**)dBarray, lddb,
                &beta,                  dCarray, lddc, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCgemmBatched(
                handle, transA, transB,
                m, n, k,
                (cuComplex*)&alpha, (const cuComplex**)dAarray, ldda, (const cuComplex**)dBarray, lddb,
                (cuComplex*)&beta,  (      cuComplex**)dCarray, lddc, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZgemmBatched(
                handle, transA, transB,
                m, n, k,
                (cuDoubleComplex*)&alpha, (const cuDoubleComplex**)dAarray, ldda, (const cuDoubleComplex**)dBarray, lddb,
                (cuDoubleComplex*)&beta,  (      cuDoubleComplex**)dCarray, lddc, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasStrsmBatched(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                (const float**)dAarray, ldda,
                (      float**)dBarray, lddb, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasDtrsmBatched(
                handle, side, uplo, trans, diag,
                m, n, &alpha,
                (const double**)dAarray, ldda,
                (      double**)dBarray, lddb, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasCtrsmBatched(
                handle, side, uplo, trans, diag,
                m, n, (cuComplex*)&alpha,
                (const cuComplex**)dAarray, ldda,
                (      cuComplex**)dBarray, lddb, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
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
    #ifdef BLASPP_WITH_CUBLAS
        blas_cublas_call(
            cublasZtrsmBatched(
                handle, side, uplo, trans, diag,
                m, n, (cuDoubleComplex*)&alpha,
                (const cuDoubleComplex**)dAarray, ldda,
                (      cuDoubleComplex**)dBarray, lddb, batch_size ) );
    #elif defined(HAVE_ROCBLAS)
        // TODO: call rocBLAS
    #endif
}

}  // namespace device
}  // namespace blas
