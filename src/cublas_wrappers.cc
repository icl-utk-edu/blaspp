// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "device_internal.hh"

#ifdef BLAS_HAVE_CUBLAS

namespace blas {
namespace internal {

//------------------------------------------------------------------------------
/// @return the corresponding device trans constant
cublasOperation_t op2cublas(blas::Op trans)
{
    switch (trans) {
        case Op::NoTrans:   return CUBLAS_OP_N; break;
        case Op::Trans:     return CUBLAS_OP_T; break;
        case Op::ConjTrans: return CUBLAS_OP_C; break;
        default: throw blas::Error( "unknown op" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device diag constant
cublasDiagType_t diag2cublas(blas::Diag diag)
{
    switch (diag) {
        case Diag::Unit:    return CUBLAS_DIAG_UNIT;     break;
        case Diag::NonUnit: return CUBLAS_DIAG_NON_UNIT; break;
        default: throw blas::Error( "unknown diag" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device uplo constant
cublasFillMode_t uplo2cublas(blas::Uplo uplo)
{
    switch (uplo) {
        case Uplo::Upper: return CUBLAS_FILL_MODE_UPPER; break;
        case Uplo::Lower: return CUBLAS_FILL_MODE_LOWER; break;
        default: throw blas::Error( "unknown uplo" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device side constant
cublasSideMode_t side2cublas(blas::Side side)
{
    switch (side) {
        case Side::Left:  return CUBLAS_SIDE_LEFT;  break;
        case Side::Right: return CUBLAS_SIDE_RIGHT; break;
        default: throw blas::Error( "unknown side" );
    }
}

//==============================================================================
// Level 1 BLAS - Device Interfaces

//------------------------------------------------------------------------------
// asum
//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float* result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSasum(
            queue.handle(),
            n,
            dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    double const* dx, device_blas_int incdx,
    double* result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDasum(
            queue.handle(),
            n,
            dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    std::complex<float> const* dx, device_blas_int incdx,
    float* result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasScasum(
            queue.handle(),
            n,
            (const cuComplex*) dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    std::complex<double> const* dx, device_blas_int incdx,
    double* result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDzasum(
            queue.handle(),
            n,
            (const cuDoubleComplex*) dx, incdx,
            result));
}

//------------------------------------------------------------------------------
// axpy
//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    float alpha,
    float const* dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSaxpy(
            queue.handle(),
            n, &alpha,
            dx, incdx,
            dy, incdy));
}

//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    double alpha,
    double const* dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDaxpy(
            queue.handle(),
            n, &alpha,
            dx, incdx,
            dy, incdy));
}

//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCaxpy(
            queue.handle(),
            n, (const cuComplex*) &alpha,
            (const cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy));
}

//------------------------------------------------------------------------------
void axpy(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZaxpy(
            queue.handle(),
            n, (const cuDoubleComplex*) &alpha,
            (const cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy));
}

//------------------------------------------------------------------------------
// dot
//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float const *dy, device_blas_int incdy,
    float *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSdot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            result));
}

//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double const *dy, device_blas_int incdy,
    double *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDdot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            result));
}

//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCdotc(
            queue.handle(),
            n,
            (const cuComplex*) dx, incdx,
            (const cuComplex*) dy, incdy,
            (cuComplex*) result));
}

//------------------------------------------------------------------------------
void dot(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZdotc(
            queue.handle(),
            n,
            (const cuDoubleComplex*) dx, incdx,
            (const cuDoubleComplex*) dy, incdy,
            (cuDoubleComplex*) result));
}

//------------------------------------------------------------------------------
void dotu(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCdotu(
            queue.handle(),
            n,
            (const cuComplex*) dx, incdx,
            (const cuComplex*) dy, incdy,
            (cuComplex*) result));
}

//------------------------------------------------------------------------------
void dotu(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZdotu(
            queue.handle(),
            n,
            (const cuDoubleComplex*) dx, incdx,
            (const cuDoubleComplex*) dy, incdy,
            (cuDoubleComplex*) result));
}

//------------------------------------------------------------------------------
// iamax
//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    float const *dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    #if CUBLAS_VER_MAJOR >= 12
        blas_dev_call(
            cublasIsamax_64(
                queue.handle(),
                n,
                dx, incdx,
                result));
    #else
        // clear memory
        if (! is_devptr( result, queue )) {
            *result = 0;
        }
        else {
            blas_dev_call(
                cudaMemsetAsync( result, 0, sizeof( *result ) ));
        }
        // convert arguments
        device_blas_int n_    = to_device_blas_int( n );
        device_blas_int incdx_ = to_device_blas_int( incdx );
        device_blas_int* result_ = (device_blas_int*) result;
        blas_dev_call(
            cublasIsamax(
                queue.handle(),
                n_,
                dx, incdx_,
                result_ ));
    #endif
    // cublas returns 1-based index
    if (! is_devptr( result, queue )) {
        *result -= 1;
    }
    else {
        shift_vec( 1, result, (int64_t) -1, queue );
    }
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    double const *dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    #if CUBLAS_VER_MAJOR >= 12
        blas_dev_call(
            cublasIdamax_64(
                queue.handle(),
                n,
                dx, incdx,
                result));
    #else
        // clear memory
        if (! is_devptr( result, queue )) {
            *result = 0;
        }
        else {
            blas_dev_call(
                cudaMemsetAsync( result, 0, sizeof( *result ) ));
        }
        // convert arguments
        device_blas_int n_    = to_device_blas_int( n );
        device_blas_int incdx_ = to_device_blas_int( incdx );
        device_blas_int* result_ = (device_blas_int*) result;

        blas_dev_call(
            cublasIdamax(
                queue.handle(),
                n_,
                dx, incdx_,
                result_));
    #endif
    // cublas returns 1-based index
    if (! is_devptr( result, queue )) {
        *result -= 1;
    }
    else {
        shift_vec( 1, result, (int64_t) -1, queue );
    }
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    std::complex<float> const *dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    #if CUBLAS_VER_MAJOR >= 12
        blas_dev_call(
            cublasIcamax_64(
                queue.handle(),
                n,
                (const cuComplex*) dx, incdx,
                result));
    #else
        // clear memory
        if (! is_devptr( result, queue )) {
            *result = 0;
        }
        else {
            blas_dev_call(
                cudaMemsetAsync( result, 0, sizeof( *result ) ));
        }
        // convert arguments
        device_blas_int n_    = to_device_blas_int( n );
        device_blas_int incdx_ = to_device_blas_int( incdx );
        device_blas_int* result_ = (device_blas_int*) result;
        blas_dev_call(
            cublasIcamax(
                queue.handle(),
                n_,
                (const cuComplex*) dx, incdx_,
                result_));
    #endif
    // cublas returns 1-based index
    if (! is_devptr( result, queue )) {
        *result -= 1;
    }
    else {
        shift_vec( 1, result, (int64_t) -1, queue );
    }
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    std::complex<double> const *dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    #if CUBLAS_VER_MAJOR >= 12
        blas_dev_call(
            cublasIzamax_64(
                queue.handle(),
                n,
                (const cuDoubleComplex*) dx, incdx,
                result));
    #else
        // clear memory
        if (! is_devptr( result, queue )) {
            *result = 0;
        }
        else {
            blas_dev_call(
                cudaMemsetAsync( result, 0, sizeof( *result ) ));
        }
        // convert arguments
        device_blas_int n_    = to_device_blas_int( n );
        device_blas_int incdx_ = to_device_blas_int( incdx );
        device_blas_int* result_ = (device_blas_int*) result;
        blas_dev_call(
            cublasIzamax(
                queue.handle(),
                n_,
                (const cuDoubleComplex*) dx, incdx_,
                result_));
    #endif
    // cublas returns 1-based index
    if (! is_devptr( result, queue )) {
        *result -= 1;
    }
    else {
        shift_vec( 1, result, (int64_t) -1, queue );
    }
}

// -----------------------------------------------------------------------------
// nrm2
//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSnrm2(
            queue.handle(),
            n, dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    double const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDnrm2(
            queue.handle(),
            n, dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    std::complex<float> const* dx, device_blas_int incdx,
    float *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasScnrm2(
            queue.handle(),
            n, (const cuComplex*) dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void nrm2(
    device_blas_int n,
    std::complex<double> const* dx, device_blas_int incdx,
    double *result,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDznrm2(
            queue.handle(),
            n, (const cuDoubleComplex*) dx, incdx,
            result));
}

//------------------------------------------------------------------------------
// scal
//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSscal(
            queue.handle(),
            n, &alpha,
            dx, incdx));
}

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDscal(
            queue.handle(),
            n, &alpha,
            dx, incdx));
}

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCscal(
            queue.handle(),
            n, (const cuComplex*) &alpha,
            (cuComplex*) dx, incdx));
}

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZscal(
            queue.handle(),
            n, (const cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dx, incdx));
}

//------------------------------------------------------------------------------
// swap
//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSswap(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDswap(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCswap(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy) );
}

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZswap(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy) );
}

//------------------------------------------------------------------------------
// copy
//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasScopy(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDcopy(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCcopy(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy) );
}

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZcopy(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy) );
}

//------------------------------------------------------------------------------
// rot
//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    float* dx, device_blas_int incdx,
    float* dy, device_blas_int incdy,
    const float c,
    const float s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSrot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            &c, &s));
}

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    double* dx, device_blas_int incdx,
    double* dy, device_blas_int incdy,
    const double c,
    const double s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDrot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            &c, &s));
}

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    std::complex<float>* dx, device_blas_int incdx,
    std::complex<float>* dy, device_blas_int incdy,
    const float c,
    const float s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCsrot(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy,
            &c, &s));
}

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    std::complex<double>* dx, device_blas_int incdx,
    std::complex<double>* dy, device_blas_int incdy,
    const double c,
    const double s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZdrot(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy,
            &c, &s));
}

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    std::complex<float>* dx, device_blas_int incdx,
    std::complex<float>* dy, device_blas_int incdy,
    const float c,
    const std::complex<float> s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCrot(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy,
            &c, (cuComplex*) &s));
}

//------------------------------------------------------------------------------
void rot(
    device_blas_int n,
    std::complex<double>* dx, device_blas_int incdx,
    std::complex<double>* dy, device_blas_int incdy,
    const double c,
    const std::complex<double> s,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZrot(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy,
            &c, (cuDoubleComplex*) &s));
}

//------------------------------------------------------------------------------
// rotg
//------------------------------------------------------------------------------
void rotg(
    float* da,
    float* db,
    float* dc,
    float* ds,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSrotg(
            queue.handle(),
            da,
            db,
            dc,
            ds));
}

//------------------------------------------------------------------------------
void rotg(
    double* da,
    double* db,
    double* dc,
    double* ds,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDrotg(
            queue.handle(),
            da,
            db,
            dc,
            ds));
}

//------------------------------------------------------------------------------
void rotg(
    std::complex<float>* da,
    std::complex<float>* db,
    float* dc,
    std::complex<float>* ds,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCrotg(
            queue.handle(),
            (cuComplex*) da,
            (cuComplex*) db,
            dc,
            (cuComplex*) ds));
}

//------------------------------------------------------------------------------
void rotg(
    std::complex<double>* da,
    std::complex<double>* db,
    double* dc,
    std::complex<double>* ds,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZrotg(
            queue.handle(),
            (cuDoubleComplex*) da,
            (cuDoubleComplex*) db,
            dc,
            (cuDoubleComplex*) ds));
}

//------------------------------------------------------------------------------
// rotm
//------------------------------------------------------------------------------
void rotm(
    device_blas_int n,
    float* dx, device_blas_int incdx,
    float* dy, device_blas_int incdy,
    const float* param,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSrotm(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            param));
}

//------------------------------------------------------------------------------
void rotm(
    device_blas_int n,
    double* dx, device_blas_int incdx,
    double* dy, device_blas_int incdy,
    const double* param,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDrotm(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            param));
}

//------------------------------------------------------------------------------
// rotmg
//------------------------------------------------------------------------------
void rotmg(
    float* d1,
    float* d2,
    float* x1,
    float* y1,
    float* param,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSrotmg(
            queue.handle(),
            d1,
            d2,
            x1,
            y1,
            param));
}

//------------------------------------------------------------------------------
void rotmg(
    double* d1,
    double* d2,
    double* x1,
    double* y1,
    double* param,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDrotmg(
            queue.handle(),
            d1,
            d2,
            x1,
            y1,
            param));
}

//==============================================================================
// Level 2 BLAS - Device Interfaces

//------------------------------------------------------------------------------
// gemv
//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const* dA, device_blas_int ldda,
    float const* dx, device_blas_int incdx,
    float beta,
    float*       dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSgemv(
            queue.handle(),
            op2cublas( trans ),
            m, n,
            &alpha, dA, ldda,
                    dx, incdx,
            &beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const* dA, device_blas_int ldda,
    double const* dx, device_blas_int incdx,
    double beta,
    double*       dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDgemv(
            queue.handle(),
            op2cublas( trans ),
            m, n,
            &alpha, dA, ldda,
                    dx, incdx,
            &beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const* dA, device_blas_int ldda,
    std::complex<float> const* dx, device_blas_int incdx,
    std::complex<float> beta,
    std::complex<float>*       dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCgemv(
            queue.handle(),
            op2cublas( trans ),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dx, incdx,
            (cuComplex*) &beta,
            (cuComplex*) dy, incdy ) );
}

//------------------------------------------------------------------------------
void gemv(
    blas::Op trans,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const* dA, device_blas_int ldda,
    std::complex<double> const* dx, device_blas_int incdx,
    std::complex<double> beta,
    std::complex<double>*       dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZgemv(
            queue.handle(),
            op2cublas( trans ),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dy, incdy ) );
}

//==============================================================================
// Level 3 BLAS - Device Interfaces

//------------------------------------------------------------------------------
// gemm
//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// trsm
//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasStrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb ) );
}

//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb ) );
}

//------------------------------------------------------------------------------
void trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb ) );
}

//------------------------------------------------------------------------------
// trmm
//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasStrmm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb,
            dB, lddb ) );
}

//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDtrmm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            dA, ldda,
            dB, lddb,
            dB, lddb ) );
}

//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCtrmm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) dB, lddb ) );
}

//------------------------------------------------------------------------------
void trmm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZtrmm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            (cuDoubleComplex*) dB, lddb ) );
}

//------------------------------------------------------------------------------
// hemm
//------------------------------------------------------------------------------
void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasChemm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void hemm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZhemm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// symm
//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void symm(
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// herk
//------------------------------------------------------------------------------
void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCherk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, (cuComplex*) dA, ldda,
            &beta,  (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void herk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZherk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, (cuDoubleComplex*) dA, ldda,
            &beta,  (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// syrk
//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void syrk(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// her2k
//------------------------------------------------------------------------------
void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCher2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void her2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZher2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// syr2k
//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
}

//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
void syr2k(
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb,
            (cuDoubleComplex*) &beta,
            (cuDoubleComplex*) dC, lddc ) );
}

//------------------------------------------------------------------------------
// batch gemm
//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasSgemmBatched(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha,
            (float const**) dAarray, ldda,
            (float const**) dBarray, lddb,
            &beta,
            dCarray, lddc,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDgemmBatched(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha,
            (double const**) dAarray, ldda,
            (double const**) dBarray, lddb,
            &beta,
            dCarray, lddc,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCgemmBatched(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            (cuComplex*)        &alpha,
            (cuComplex const**) dAarray, ldda,
            (cuComplex const**) dBarray, lddb,
            (cuComplex*)        &beta,
            (cuComplex**)       dCarray, lddc,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZgemmBatched(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            (cuDoubleComplex*)        &alpha,
            (cuDoubleComplex const**) dAarray, ldda,
            (cuDoubleComplex const**) dBarray, lddb,
            (cuDoubleComplex*)        &beta,
            (cuDoubleComplex**)       dCarray, lddc,
            batch_size ) );
}

//------------------------------------------------------------------------------
// batch trsm
//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasStrsmBatched(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            (float const**) dAarray, ldda,
            (float**)       dBarray, lddb,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasDtrsmBatched(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
            (double const**) dAarray, ldda,
            (double**)       dBarray, lddb,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasCtrsmBatched(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuComplex*)        &alpha,
            (cuComplex const**) dAarray, ldda,
            (cuComplex**)       dBarray, lddb,
            batch_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size,
    blas::Queue& queue )
{
    blas_dev_call(
        cublasZtrsmBatched(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuDoubleComplex*)        &alpha,
            (cuDoubleComplex const**) dAarray, ldda,
            (cuDoubleComplex**)       dBarray, lddb,
            batch_size ) );
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS_HAVE_CUBLAS
