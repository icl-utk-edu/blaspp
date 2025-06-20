// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "device_internal.hh"

#ifdef BLAS_HAVE_SYCL

// see https://software.intel.com/content/www/us/en/develop/articles/use-of-intel-mkl-data-types-in-cc-applications.html
#include "blas/config.h"
#define MKL_Complex8  blas_complex_float
#define MKL_Complex16 blas_complex_double
#include <oneapi/mkl.hpp>

namespace blas {
namespace internal {

//------------------------------------------------------------------------------
/// @return the corresponding device trans constant
oneapi::mkl::transpose op2onemkl(blas::Op trans)
{
    switch (trans) {
        case Op::NoTrans:   return oneapi::mkl::transpose::N; break;
        case Op::Trans:     return oneapi::mkl::transpose::T; break;
        case Op::ConjTrans: return oneapi::mkl::transpose::C; break;
        default: throw blas::Error( "unknown op" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device diag constant
oneapi::mkl::diag diag2onemkl(blas::Diag diag)
{
    switch (diag) {
        case Diag::Unit:    return oneapi::mkl::diag::U; break;
        case Diag::NonUnit: return oneapi::mkl::diag::N; break;
        default: throw blas::Error( "unknown diag" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device uplo constant
oneapi::mkl::uplo uplo2onemkl(blas::Uplo uplo)
{
    switch (uplo) {
        case Uplo::Upper: return oneapi::mkl::uplo::U; break;
        case Uplo::Lower: return oneapi::mkl::uplo::L; break;
        default: throw blas::Error( "unknown uplo" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device side constant
oneapi::mkl::side side2onemkl(blas::Side side)
{
    switch (side) {
        case Side::Left:  return oneapi::mkl::side::L;  break;
        case Side::Right: return oneapi::mkl::side::R; break;
        default: throw blas::Error( "unknown side" );
    }
}

//==============================================================================
// Level 1 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
// asum
//------------------------------------------------------------------------------
void asum(
    device_blas_int n,
    float const* dx, device_blas_int incdx,
    float* result,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::asum(
            queue.stream(),
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
        oneapi::mkl::blas::asum(
            queue.stream(),
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
        oneapi::mkl::blas::asum(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::asum(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::axpy(
            queue.stream(),
            n,
            alpha,
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
        oneapi::mkl::blas::axpy(
            queue.stream(),
            n,
            alpha,
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
        oneapi::mkl::blas::axpy(
            queue.stream(),
            n,
            alpha,
            dx, incdx,
            dy, incdy));
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
        oneapi::mkl::blas::axpy(
            queue.stream(),
            n,
            alpha,
            dx, incdx,
            dy, incdy));
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
        oneapi::mkl::blas::dot(
            queue.stream(),
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
        oneapi::mkl::blas::dot(
            queue.stream(),
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
        oneapi::mkl::blas::dotc(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            result));
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
        oneapi::mkl::blas::dotc(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            result));
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
        oneapi::mkl::blas::dotu(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            result));
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
        oneapi::mkl::blas::dotu(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            result));
}

//------------------------------------------------------------------------------
// iamax
//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    float const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // Return -1 for non-positive n or incx. MKL routine would return 0.
    if (n < 1 || incdx < 1) {
        if (is_devptr( result, queue )) {
            int64_t tmp = -1;
            device_memcpy( result, &tmp, 1, queue );
            queue.sync();
        }
        else {
            *result = -1;
        }
        return;
    }
    blas_dev_call(
        oneapi::mkl::blas::iamax(
            queue.stream(),
            n,
            dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    double const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // Return -1 for non-positive n or incx. MKL routine would return 0.
    if (n < 1 || incdx < 1) {
        if (is_devptr( result, queue )) {
            int64_t tmp = -1;
            device_memcpy( result, &tmp, 1, queue );
            queue.sync();
        }
        else {
            *result = -1;
        }
        return;
    }
    blas_dev_call(
        oneapi::mkl::blas::iamax(
            queue.stream(),
            n,
            dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    std::complex<float> const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // Return -1 for non-positive n or incx. MKL routine would return 0.
    if (n < 1 || incdx < 1) {
        if (is_devptr( result, queue )) {
            int64_t tmp = -1;
            device_memcpy( result, &tmp, 1, queue );
            queue.sync();
        }
        else {
            *result = -1;
        }
        return;
    }
    blas_dev_call(
        oneapi::mkl::blas::iamax(
            queue.stream(),
            n,
            dx, incdx,
            result));
}

//------------------------------------------------------------------------------
void iamax(
    int64_t n,
    std::complex<double> const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // Return -1 for non-positive n or incx. MKL routine would return 0.
    if (n < 1 || incdx < 1) {
        if (is_devptr( result, queue )) {
            int64_t tmp = -1;
            device_memcpy( result, &tmp, 1, queue );
            queue.sync();
        }
        else {
            *result = -1;
        }
        return;
    }
    blas_dev_call(
        oneapi::mkl::blas::iamax(
            queue.stream(),
            n,
            dx, incdx,
            result));
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
        oneapi::mkl::blas::nrm2(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::nrm2(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::nrm2(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::nrm2(
            queue.stream(),
            n,
            dx, incdx,
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
        oneapi::mkl::blas::scal(
            queue.stream(),
            n,
            alpha,
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
        oneapi::mkl::blas::scal(
            queue.stream(),
            n,
            alpha,
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
        oneapi::mkl::blas::scal(
            queue.stream(),
            n,
            alpha,
            dx, incdx));
}

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::scal(
            queue.stream(),
            n,
            alpha,
            dx, incdx));
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
        oneapi::mkl::blas::swap(
            queue.stream(),
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
        oneapi::mkl::blas::swap(
            queue.stream(),
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
        oneapi::mkl::blas::swap(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::swap(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy) );
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
        oneapi::mkl::blas::copy(
            queue.stream(),
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
        oneapi::mkl::blas::copy(
            queue.stream(),
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
        oneapi::mkl::blas::copy(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy) );
}

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::copy(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy) );
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rot(
            queue.stream(),
            n,
            dx, incdx,
            dy, incdy,
            c, s));
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
        oneapi::mkl::blas::rotg(
            queue.stream(),
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
        oneapi::mkl::blas::rotg(
            queue.stream(),
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
        oneapi::mkl::blas::rotg(
            queue.stream(),
            da,
            db,
            dc,
            ds));
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
        oneapi::mkl::blas::rotg(
            queue.stream(),
            da,
            db,
            dc,
            ds));
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
        oneapi::mkl::blas::rotm(
            queue.stream(),
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
        oneapi::mkl::blas::rotm(
            queue.stream(),
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
        oneapi::mkl::blas::rotmg(
            queue.stream(),
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
        oneapi::mkl::blas::rotmg(
            queue.stream(),
            d1,
            d2,
            x1,
            y1,
            param));
}

//==============================================================================
// Level 2 BLAS - Device Interfaces

//------------------------------------------------------------------------------
// hemv
//------------------------------------------------------------------------------
void hemv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* dA, int64_t ldda,
    std::complex<float> const* dx, int64_t incdx,
    std::complex<float> beta,
    std::complex<float>*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::hemv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void hemv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* dA, int64_t ldda,
    std::complex<double> const* dx, int64_t incdx,
    std::complex<double> beta,
    std::complex<double>*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::hemv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
// symv
//------------------------------------------------------------------------------
void symv(
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const* dA, int64_t ldda,
    float const* dx, int64_t incdx,
    float beta,
    float*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::symv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void symv(
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const* dA, int64_t ldda,
    double const* dx, int64_t incdx,
    double beta,
    double*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::symv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void symv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const* dA, int64_t ldda,
    std::complex<float> const* dx, int64_t incdx,
    std::complex<float> beta,
    std::complex<float>*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::symv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
}

//------------------------------------------------------------------------------
void symv(
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const* dA, int64_t ldda,
    std::complex<double> const* dx, int64_t incdx,
    std::complex<double> beta,
    std::complex<double>*       dy, int64_t incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        oneapi::mkl::blas::symv(
            queue.stream(),
            uplo2cublas( uplo ),
            n,
            alpha, dA, ldda,
                   dx, incdx,
            beta,  dy, incdy ) );
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
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
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
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
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
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
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
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
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
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
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
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
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
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
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
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
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
        oneapi::mkl::blas::hemm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::hemm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::herk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::herk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::her2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::her2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
         oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
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
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const float**)dAarray, &ldda,
            (const float**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
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
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const double**)dAarray, &ldda,
            (const double**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
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
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const std::complex<float>**)dAarray, &ldda,
            (const std::complex<float>**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
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
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const std::complex<double>**)dAarray, &ldda,
            (const std::complex<double>**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
}

//------------------------------------------------------------------------------
// batch gemm, group API
//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    float *alpha,
    float const * const * dAarray, device_blas_int *ldda,
    float const * const * dBarray, device_blas_int *lddb,
    float *beta,
    float** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for (auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const float**)dAarray, ldda,
            (const float**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    double *alpha,
    double const * const * dAarray, device_blas_int *ldda,
    double const * const * dBarray, device_blas_int *lddb,
    double *beta,
    double** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for (auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const double**)dAarray, ldda,
            (const double**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    std::complex<float> *alpha,
    std::complex<float> const * const * dAarray, device_blas_int *ldda,
    std::complex<float> const * const * dBarray, device_blas_int *lddb,
    std::complex<float> *beta,
    std::complex<float>** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for (auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const std::complex<float>**)dAarray, ldda,
            (const std::complex<float>**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_gemm(
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    std::complex<double> *alpha,
    std::complex<double> const * const * dAarray, device_blas_int *ldda,
    std::complex<double> const * const * dBarray, device_blas_int *lddb,
    std::complex<double> *beta,
    std::complex<double>** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for (auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const std::complex<double>**)dAarray, ldda,
            (const std::complex<double>**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
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
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            &side_, &uplo_, &trans_, &diag_,
            &m, &n,
            &alpha,
            (const float**)dAarray, &ldda,
            (      float**)dBarray, &lddb,
            1, &batch_size ) );
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
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            &side_, &uplo_, &trans_, &diag_,
            &m, &n,
            &alpha,
            (const double**)dAarray, &ldda,
            (      double**)dBarray, &lddb,
            1, &batch_size ) );
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
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            &side_, &uplo_, &trans_, &diag_,
            &m, &n,
            &alpha,
            (const std::complex<float>**)dAarray, &ldda,
            (      std::complex<float>**)dBarray, &lddb,
            1, &batch_size ) );
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
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            &side_, &uplo_, &trans_, &diag_,
            &m, &n,
            &alpha,
            (const std::complex<double>**)dAarray, &ldda,
            (      std::complex<double>**)dBarray, &lddb,
            1, &batch_size ) );
}

//------------------------------------------------------------------------------
// batch trsm, group API
//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    float *alpha,
    float const * const * dAarray, device_blas_int *ldda,
    float const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for (auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            side_.data(), uplo_.data(), trans_.data(), diag_.data(),
            m, n,
            alpha,
            (const float**)dAarray, ldda,
            (      float**)dBarray, lddb,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    double *alpha,
    double const * const * dAarray, device_blas_int *ldda,
    double const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for (auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            side_.data(), uplo_.data(), trans_.data(), diag_.data(),
            m, n,
            alpha,
            (const double**)dAarray, ldda,
            (      double**)dBarray, lddb,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    std::complex<float> *alpha,
    std::complex<float> const * const * dAarray, device_blas_int *ldda,
    std::complex<float> const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for (auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            side_.data(), uplo_.data(), trans_.data(), diag_.data(),
            m, n,
            alpha,
            (const std::complex<float>**)dAarray, ldda,
            (      std::complex<float>**)dBarray, lddb,
            group_count, group_size ) );
}

//------------------------------------------------------------------------------
void batch_trsm(
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    std::complex<double> *alpha,
    std::complex<double> const * const * dAarray, device_blas_int *ldda,
    std::complex<double> const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size,
    blas::Queue& queue )
{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for (auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
            queue.stream(),
            side_.data(), uplo_.data(), trans_.data(), diag_.data(),
            m, n,
            alpha,
            (const std::complex<double>**)dAarray, ldda,
            (      std::complex<double>**)dBarray, lddb,
            group_count, group_size ) );
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS_HAVE_SYCL
