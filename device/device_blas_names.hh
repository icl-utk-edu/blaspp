#ifndef DEVICE_BLAS_HH
#define DEVICE_BLAS_HH

// =============================================================================
// Level 1 BLAS - Device Interfaces

// -----------------------------------------------------------------------------

// =============================================================================
// Level 2 BLAS - Device Interfaces

// -----------------------------------------------------------------------------

// =============================================================================
// Level 3 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
void DEVICE_BLAS_sgemm( 
    device_blas_handle_t handle, 
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc)
{
    #ifdef HAVE_CUBLAS
    cublasSgemm( handle, transA, transB, 
                 m, n, k, 
                 &alpha, dA, ldda, dB, lddb, 
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

void DEVICE_BLAS_dgemm( 
    device_blas_handle_t handle, 
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc)
{
    #ifdef HAVE_CUBLAS
    cublasDgemm( handle, transA, transB, 
                 m, n, k, 
                 &alpha, dA, ldda, dB, lddb, 
                 &beta,  dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

void DEVICE_BLAS_cgemm( 
    device_blas_handle_t handle, 
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc)
{
    #ifdef HAVE_CUBLAS
    cublasCgemm( handle, transA, transB, m, n, k, 
                (cuComplex*)&alpha, (cuComplex*)dA, ldda, (cuComplex*)dB, lddb, 
                (cuComplex*)&beta,  (cuComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

void DEVICE_BLAS_zgemm( 
    device_blas_handle_t handle, 
    device_trans_t transA, device_trans_t transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc)
{
    #ifdef HAVE_CUBLAS
    cublasZgemm( handle, transA, transB, 
                 m, n, k, 
                 (cuDoubleComplex*)&alpha, (cuDoubleComplex*)dA, ldda, (cuDoubleComplex*)dB, lddb, 
                 (cuDoubleComplex*)&beta,  (cuDoubleComplex*)dC, lddc );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocBLAS
    #endif
}

#endif        //  #ifndef DEVICE_BLAS_HH

