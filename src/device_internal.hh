// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_DEVICE_NAMES_HH
#define BLAS_DEVICE_NAMES_HH

#include "blas/device.hh"

#include <complex>

namespace blas {

//------------------------------------------------------------------------------
/// @see to_device_blas_int
///
inline device_blas_int to_device_blas_int_( int64_t x, const char* x_str )
{
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if_msg( std::abs( x ) > std::numeric_limits<device_blas_int>::max(),
                           "%s", x_str );
    }
    return device_blas_int( x );
}

//----------------------------------------
/// Convert int64_t to device_blas_int.
/// If device_blas_int is 64-bit, this does nothing.
/// If device_blas_int is 32-bit, throws if x > INT_MAX, so conversion would overflow.
///
/// Note this is in src/device_internal.hh, so this macro won't pollute
/// the namespace when apps #include <blas.hh>.
///
#define to_device_blas_int( x ) to_device_blas_int_( x, #x )

namespace internal {

//==============================================================================
// Level 1 BLAS - Device Interfaces
// Alphabetical order

//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    float alpha,
    float const* dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    double alpha,
    double const* dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void axpy(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float const *dy, device_blas_int incdy,
    float *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double const *dy, device_blas_int incdy,
    double *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue );

void dot(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue );

void dotu(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue );

void dotu(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    double const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    std::complex<float> const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue );

void nrm2(
    device_blas_int n,
    std::complex<double> const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx,
    blas::Queue& queue );

void scal(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void swap(
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue );

void copy(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue );


//==============================================================================
// Level 2 BLAS - Device Interfaces

//------------------------------------------------------------------------------

//==============================================================================
// Level 3 BLAS - Device Interfaces

//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc,
    blas::Queue& queue );

void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue );

void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue );

void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch gemm
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue );

//------------------------------------------------------------------------------
// batch trsm
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue );

}  // namespace internal
}  // namespace blas

#endif        //  #ifndef BLAS_DEVICE_NAMES_HH
