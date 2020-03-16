// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

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
void DEVICE_sgemm(
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
    cublasSgemm( handle, transA, transB,
                 m, n, k,
                 &alpha, dA, ldda, dB, lddb,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// dgemm
void DEVICE_dgemm(
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
    cublasDgemm( handle, transA, transB,
                 m, n, k,
                 &alpha, dA, ldda, dB, lddb,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// cgemm
void DEVICE_cgemm(
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
    cublasCgemm( handle, transA, transB, m, n, k,
                (cuComplex*)&alpha, (cuComplex*)dA, ldda, (cuComplex*)dB, lddb,
                (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// zgemm
void DEVICE_zgemm(
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
    cublasZgemm( handle, transA, transB,
                 m, n, k,
                 (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda, (cuDoubleComplex*)dB, lddb,
                 (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// trsm
// -----------------------------------------------------------------------------
// strsm
void DEVICE_strsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasStrsm( handle, side, uplo, trans, diag,
                 m, n, &alpha,
                 dA, ldda,
                 dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// dtrsm
void DEVICE_dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasDtrsm( handle, side, uplo, trans, diag,
                 m, n, &alpha,
                 dA, ldda,
                 dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// ctrsm
void DEVICE_ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasCtrsm( handle, side, uplo, trans, diag,
                 m, n, (cuComplex*)&alpha,
                 (cuComplex*)dA, ldda,
                 (cuComplex*)dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// ztrsm
void DEVICE_ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasZtrsm( handle, side, uplo, trans, diag,
                 m, n, (cuDoubleComplex*)&alpha,
                 (cuDoubleComplex*)dA, ldda,
                 (cuDoubleComplex*)dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// trmm
// -----------------------------------------------------------------------------
// strmm
void DEVICE_strmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasStrmm( handle, side, uplo, trans, diag,
                 m, n, &alpha,
                 dA, ldda,
                 dB, lddb,
                 dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// dtrmm
void DEVICE_dtrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasDtrmm( handle, side, uplo, trans, diag,
                 m, n, &alpha,
                 dA, ldda,
                 dB, lddb,
                 dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// ctrmm
void DEVICE_ctrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasCtrmm( handle, side, uplo, trans, diag,
                 m, n, (cuComplex*)&alpha,
                 (cuComplex*)dA, ldda,
                 (cuComplex*)dB, lddb,
                 (cuComplex*)dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// ztrmm
void DEVICE_ztrmm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasZtrmm( handle, side, uplo, trans, diag,
                 m, n, (cuDoubleComplex*)&alpha,
                 (cuDoubleComplex*)dA, ldda,
                 (cuDoubleComplex*)dB, lddb,
                 (cuDoubleComplex*)dB, lddb );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// hemm
// -----------------------------------------------------------------------------
// chemm
void DEVICE_chemm(
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
    cublasChemm( handle, side, uplo,
                 m, n,
                 (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                                     (cuComplex*)dB, lddb,
                 (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zhemm
void DEVICE_zhemm(
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
    cublasZhemm( handle, side, uplo,
                 m, n,
                 (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                                           (cuDoubleComplex*)dB, lddb,
                 (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// symm
// -----------------------------------------------------------------------------
// ssymm
void DEVICE_ssymm(
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
    cublasSsymm( handle, side, uplo,
                 m, n,
                 &alpha, dA, ldda,
                         dB, lddb,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// dsymm
void DEVICE_dsymm(
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
    cublasDsymm( handle, side, uplo,
                 m, n,
                 &alpha, dA, ldda,
                         dB, lddb,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// csymm
void DEVICE_csymm(
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
    cublasCsymm( handle, side, uplo,
                 m, n,
                 (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                                     (cuComplex*)dB, lddb,
                 (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zsymm
void DEVICE_zsymm(
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
    cublasZsymm( handle, side, uplo,
                 m, n,
                 (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                                           (cuDoubleComplex*)dB, lddb,
                 (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// herk
// -----------------------------------------------------------------------------
// cherk
void DEVICE_cherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasCherk( handle, uplo, trans,
                 n, k,
                 &alpha, (cuComplex*)dA, ldda,
                 &beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zherk
void DEVICE_zherk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasZherk( handle, uplo, trans,
                 n, k,
                 &alpha, (cuDoubleComplex*)dA, ldda,
                 &beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// syrk
// -----------------------------------------------------------------------------
// ssyrk
void DEVICE_ssyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasSsyrk( handle, uplo, trans,
                 n, k,
                 &alpha, dA, ldda,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// dsyrk
void DEVICE_dsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasDsyrk( handle, uplo, trans,
                 n, k,
                 &alpha, dA, ldda,
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// csyrk
void DEVICE_csyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasCsyrk( handle, uplo, trans,
                 n, k,
                 (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                 (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zsyrk
void DEVICE_zsyrk(
    device_blas_handle_t handle,
    device_uplo_t uplo, device_trans_t trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasZsyrk( handle, uplo, trans,
                 n, k,
                 (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                 (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// her2k
// -----------------------------------------------------------------------------
// cher2k
void DEVICE_cher2k(
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
    cublasCher2k( handle, uplo, trans,
                  n, k,
                  (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                                      (cuComplex*)dB, lddb,
                  &beta,              (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zher2k
void DEVICE_zher2k(
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
    cublasZher2k( handle, uplo, trans,
                  n, k,
                  (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                                            (cuDoubleComplex*)dB, lddb,
                  &beta,                    (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// syr2k
// -----------------------------------------------------------------------------
// ssyr2k
void DEVICE_ssyr2k(
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
    cublasSsyr2k( handle, uplo, trans,
                  n, k,
                  &alpha, dA, ldda,
                          dB, lddb,
                  &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// dsyr2k
void DEVICE_dsyr2k(
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
    cublasDsyr2k( handle, uplo, trans,
                  n, k,
                  &alpha, dA, ldda,
                          dB, lddb,
                  &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// csyr2k
void DEVICE_csyr2k(
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
    cublasCsyr2k( handle, uplo, trans,
                  n, k,
                  (cuComplex*)&alpha, (cuComplex*)dA, ldda,
                                      (cuComplex*)dB, lddb,
                  (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// zsyr2k
void DEVICE_zsyr2k(
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
    cublasZsyr2k( handle, uplo, trans,
                  n, k,
                  (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda,
                                            (cuDoubleComplex*)dB, lddb,
                  (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif

}

// -----------------------------------------------------------------------------
// batch gemm
// -----------------------------------------------------------------------------
// batch sgemm
void DEVICE_BATCH_sgemm(
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
    cublasSgemmBatched( handle, transA, transB,
                        m, n, k,
                        &alpha, (const float**)dAarray, ldda, (const float**)dBarray, lddb,
                        &beta,                 dCarray, lddc, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch dgemm
void DEVICE_BATCH_dgemm(
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
    cublasDgemmBatched( handle, transA, transB,
                        m, n, k,
                        &alpha, (const double**)dAarray, ldda, (const double**)dBarray, lddb,
                        &beta,                  dCarray, lddc, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch cgemm
void DEVICE_BATCH_cgemm(
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
    cublasCgemmBatched( handle, transA, transB,
                        m, n, k,
                        (cuComplex*)&alpha, (const cuComplex**)dAarray, ldda, (const cuComplex**)dBarray, lddb,
                        (cuComplex*)&beta,  (      cuComplex**)dCarray, lddc, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch zgemm
void DEVICE_BATCH_zgemm(
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
    cublasZgemmBatched( handle, transA, transB,
                        m, n, k,
                        (cuDoubleComplex*)&alpha, (const cuDoubleComplex**)dAarray, ldda, (const cuDoubleComplex**)dBarray, lddb,
                        (cuDoubleComplex*)&beta,  (      cuDoubleComplex**)dCarray, lddc, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch strsm
void DEVICE_BATCH_strsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)

{
    #ifdef BLASPP_WITH_CUBLAS
    cublasStrsmBatched( handle, side, uplo, trans, diag,
                        m, n, &alpha,
                        (const float**)dAarray, ldda,
                        (      float**)dBarray, lddb, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch dtrsm
void DEVICE_BATCH_dtrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasDtrsmBatched( handle, side, uplo, trans, diag,
                        m, n, &alpha,
                        (const double**)dAarray, ldda,
                        (      double**)dBarray, lddb, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch ctrsm
void DEVICE_BATCH_ctrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasCtrsmBatched( handle, side, uplo, trans, diag,
                        m, n, (cuComplex*)&alpha,
                        (const cuComplex**)dAarray, ldda,
                        (      cuComplex**)dBarray, lddb, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
// batch ztrsm
void DEVICE_BATCH_ztrsm(
    device_blas_handle_t handle,
    device_side_t side, device_uplo_t uplo, device_trans_t trans, device_diag_t diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    #ifdef BLASPP_WITH_CUBLAS
    cublasZtrsmBatched( handle, side, uplo, trans, diag,
                        m, n, (cuDoubleComplex*)&alpha,
                        (const cuDoubleComplex**)dAarray, ldda,
                        (      cuDoubleComplex**)dBarray, lddb, batch_size );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}
