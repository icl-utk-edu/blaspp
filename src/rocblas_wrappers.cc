// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#define ROCBLAS_V3

#include "device_internal.hh"

#ifdef BLAS_HAVE_ROCBLAS

// ROCm doesn't define ROCBLAS_VERSION as one variable (as of 5.7.1).
#define rocblas_version ROCBLAS_VERSION_MAJOR*10000000 + ROCBLAS_VERSION_MINOR*100000 + ROCBLAS_VERSION_PATCH

#ifdef __HIP_PLATFORM_NVCC__
    #warning Compiling with rocBLAS on NVCC mode... This is an odd configuration. (Consider using NVCC only)
#endif

namespace blas {
namespace internal {

//------------------------------------------------------------------------------
/// @return the corresponding device trans constant
rocblas_operation op2rocblas(blas::Op trans)
{
    switch (trans) {
        case Op::NoTrans:   return rocblas_operation_none; break;
        case Op::Trans:     return rocblas_operation_transpose; break;
        case Op::ConjTrans: return rocblas_operation_conjugate_transpose; break;
        default: throw blas::Error( "unknown op" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device diag constant
rocblas_diagonal diag2rocblas(blas::Diag diag)
{
    switch (diag) {
        case Diag::Unit:    return rocblas_diagonal_unit;     break;
        case Diag::NonUnit: return rocblas_diagonal_non_unit; break;
        default: throw blas::Error( "unknown diag" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device uplo constant
rocblas_fill uplo2rocblas(blas::Uplo uplo)
{
    switch (uplo) {
        case Uplo::Upper:   return rocblas_fill_upper; break;
        case Uplo::Lower:   return rocblas_fill_lower; break;
        case Uplo::General: return rocblas_fill_full;  break;
        default: throw blas::Error( "unknown uplo" );
    }
}

//------------------------------------------------------------------------------
/// @return the corresponding device side constant
rocblas_side side2rocblas(blas::Side side)
{
    switch (side) {
        case Side::Left:  return rocblas_side_left;  break;
        case Side::Right: return rocblas_side_right; break;
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
        rocblas_sasum(
            queue.handle(),
            n, dx, incdx,
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
        rocblas_dasum(
            queue.handle(),
            n, dx, incdx,
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
        rocblas_scasum(
            queue.handle(),
            n, (rocblas_float_complex*) dx, incdx,
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
        rocblas_dzasum(
            queue.handle(),
            n, (rocblas_double_complex*) dx, incdx,
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
        rocblas_saxpy(
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
        rocblas_daxpy(
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
        rocblas_caxpy(
            queue.handle(),
            n, (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) dy, incdy));
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
        rocblas_zaxpy(
            queue.handle(),
            n, (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) dy, incdy));
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
        rocblas_sdot(
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
        rocblas_ddot(
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
        rocblas_cdotc(
            queue.handle(),
            n,
            (const rocblas_float_complex*) dx, incdx,
            (const rocblas_float_complex*) dy, incdy,
            (rocblas_float_complex*) result));
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
        rocblas_zdotc(
            queue.handle(),
            n,
            (const rocblas_double_complex*) dx, incdx,
            (const rocblas_double_complex*) dy, incdy,
            (rocblas_double_complex*) result));
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
        rocblas_cdotu(
            queue.handle(),
            n,
            (const rocblas_float_complex*) dx, incdx,
            (const rocblas_float_complex*) dy, incdy,
            (rocblas_float_complex*) result));
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
        rocblas_zdotu(
            queue.handle(),
            n,
            (const rocblas_double_complex*) dx, incdx,
            (const rocblas_double_complex*) dy, incdy,
            (rocblas_double_complex*) result));
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
    // clear memory
    if (! is_devptr( result, queue )) {
        *result = 0;
    }
    else {
        blas_dev_call(
            hipMemsetAsync( result, 0, sizeof( *result ) ));
    }
    // convert arguments
    device_blas_int n_ = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    device_blas_int* result_ = (device_blas_int*) result;

    blas_dev_call(
        rocblas_isamax(
            queue.handle(),
            n_, dx, incdx_,
            result_));

    // rocblas returns 1-based index
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
    double const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // clear memory
    if (! is_devptr( result, queue )) {
        *result = 0;
    }
    else {
        blas_dev_call(
            hipMemsetAsync( result, 0, sizeof( *result ) ));
    }
    // convert arguments
    device_blas_int n_ = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    device_blas_int* result_ = (device_blas_int*) result;

    blas_dev_call(
        rocblas_idamax(
            queue.handle(),
            n_, dx, incdx_,
            result_));

    // rocblas returns 1-based index
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
    std::complex<float> const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // clear memory
    if (! is_devptr( result, queue )) {
        *result = 0;
    }
    else {
        blas_dev_call(
            hipMemsetAsync( result, 0, sizeof( *result ) ));
    }
    // convert arguments
    device_blas_int n_ = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    device_blas_int* result_ = (device_blas_int*) result;

    blas_dev_call(
        rocblas_icamax(
            queue.handle(),
            n_, (rocblas_float_complex*) dx, incdx_,
            result_));

    // rocblas returns 1-based index
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
    std::complex<double> const* dx, int64_t incdx,
    int64_t* result,
    blas::Queue& queue )
{
    // clear memory
    if (! is_devptr( result, queue )) {
        *result = 0;
    }
    else {
        blas_dev_call(
            hipMemsetAsync( result, 0, sizeof( *result ) ));
    }
    // convert arguments
    device_blas_int n_ = to_device_blas_int( n );
    device_blas_int incdx_ = to_device_blas_int( incdx );
    device_blas_int* result_ = (device_blas_int*) result;

    blas_dev_call(
        rocblas_izamax(
            queue.handle(),
            n_, (rocblas_double_complex*) dx, incdx_,
            result_));

    // rocblas returns 1-based index
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
        rocblas_snrm2(
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
        rocblas_dnrm2(
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
        rocblas_scnrm2(
            queue.handle(),
            n, (rocblas_float_complex*) dx, incdx,
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
        rocblas_dznrm2(
            queue.handle(),
            n, (rocblas_double_complex*) dx, incdx,
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
        rocblas_sscal(
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
        rocblas_dscal(
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
        rocblas_cscal(
            queue.handle(),
            n, (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dx, incdx));
}

//------------------------------------------------------------------------------
void scal(
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    blas::Queue& queue )
{
    blas_dev_call(
        rocblas_zscal(
            queue.handle(),
            n, (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dx, incdx));
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
        rocblas_sswap(
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
        rocblas_dswap(
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
        rocblas_cswap(
            queue.handle(),
            n,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) dy, incdy) );
}

//------------------------------------------------------------------------------
void swap(
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        rocblas_zswap(
            queue.handle(),
            n,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) dy, incdy) );
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
        rocblas_scopy(
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
        rocblas_dcopy(
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
        rocblas_ccopy(
            queue.handle(),
            n,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) dy, incdy) );
}

//------------------------------------------------------------------------------
void copy(
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy,
    blas::Queue& queue )
{
    blas_dev_call(
        rocblas_zcopy(
            queue.handle(),
            n,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) dy, incdy) );
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
        rocblas_srot(
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
        rocblas_drot(
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
        rocblas_csrot(
            queue.handle(),
            n,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) dy, incdy,
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
        rocblas_zdrot(
            queue.handle(),
            n,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) dy, incdy,
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
        rocblas_crot(
            queue.handle(),
            n,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) dy, incdy,
            &c, (rocblas_float_complex*) &s));
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
        rocblas_zrot(
            queue.handle(),
            n,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) dy, incdy,
            &c, (rocblas_double_complex*) &s));
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
        rocblas_srotg(
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
        rocblas_drotg(
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
        rocblas_crotg(
            queue.handle(),
            (rocblas_float_complex*) da,
            (rocblas_float_complex*) db,
            dc,
            (rocblas_float_complex*) ds));
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
        rocblas_zrotg(
            queue.handle(),
            (rocblas_double_complex*) da,
            (rocblas_double_complex*) db,
            dc,
            (rocblas_double_complex*) ds));
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
        rocblas_srotm(
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
        rocblas_drotm(
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
        rocblas_srotmg(
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
        rocblas_drotmg(
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
        rocblas_ssymv(
            queue.handle(),
            uplo2rocblas( uplo ),
            n,
            &alpha, dA, ldda,
                    dx, incdx,
            &beta,  dy, incdy ) );
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
        rocblas_dsymv(
            queue.handle(),
            uplo2rocblas( uplo ),
            n,
            &alpha, dA, ldda,
                    dx, incdx,
            &beta,  dy, incdy ) );
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
        rocblas_csymv(
            queue.handle(),
            uplo2rocblas( uplo ),
            n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dx, incdx,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dy, incdy ) );
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
        rocblas_zsymv(
            queue.handle(),
            uplo2rocblas( uplo ),
            n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dx, incdx,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dy, incdy ) );
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
        rocblas_sgemm(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
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
        rocblas_dgemm(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
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
        rocblas_cgemm(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
            m, n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zgemm(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
            m, n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
        rocblas_strsm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
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
        rocblas_dtrsm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
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
        rocblas_ctrsm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb ) );
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
        rocblas_ztrsm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb ) );
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
        rocblas_strmm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            &alpha,
            dA, ldda,
            #if rocblas_version >= 30000000  // 3.0.0 (ROCm 5.6.0)
            dB, lddb,
            #endif
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
        rocblas_dtrmm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            &alpha,
            dA, ldda,
            #if rocblas_version >= 30000000  // 3.0.0 (ROCm 5.6.0)
            dB, lddb,
            #endif
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
        rocblas_ctrmm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            #if rocblas_version >= 30000000  // 3.0.0 (ROCm 5.6.0)
            (rocblas_float_complex*) dB, lddb,
            #endif
            (rocblas_float_complex*) dB, lddb ) );
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
        rocblas_ztrmm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            #if rocblas_version >= 30000000  // 3.0.0 (ROCm 5.6.0)
            (rocblas_double_complex*) dB, lddb,
            #endif
            (rocblas_double_complex*) dB, lddb ) );
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
        rocblas_chemm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zhemm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
        rocblas_ssymm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
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
        rocblas_dsymm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
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
        rocblas_csymm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
            m, n,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zsymm(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo),
            m, n,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
        rocblas_cherk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            &alpha, (rocblas_float_complex*) dA, ldda,
            &beta,  (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zherk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            &alpha, (rocblas_double_complex*) dA, ldda,
            &beta,  (rocblas_double_complex*) dC, lddc ) );
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
    // rocblas doesn't accept ConjTrans.
    if (trans == Op::ConjTrans)
        trans = Op::Trans;
    blas_dev_call(
        rocblas_ssyrk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
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
    // rocblas doesn't accept ConjTrans.
    if (trans == Op::ConjTrans)
        trans = Op::Trans;
    blas_dev_call(
        rocblas_dsyrk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
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
        rocblas_csyrk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zsyrk(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
        rocblas_cher2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zher2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
    // rocblas doesn't accept ConjTrans.
    if (trans == Op::ConjTrans)
        trans = Op::Trans;
    blas_dev_call(
        rocblas_ssyr2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
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
    // rocblas doesn't accept ConjTrans.
    if (trans == Op::ConjTrans)
        trans = Op::Trans;
    blas_dev_call(
        rocblas_dsyr2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
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
        rocblas_csyr2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_float_complex*) &alpha,
            (rocblas_float_complex*) dA, ldda,
            (rocblas_float_complex*) dB, lddb,
            (rocblas_float_complex*) &beta,
            (rocblas_float_complex*) dC, lddc ) );
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
        rocblas_zsyr2k(
            queue.handle(),
            uplo2rocblas(uplo), op2rocblas(trans),
            n, k,
            (rocblas_double_complex*) &alpha,
            (rocblas_double_complex*) dA, ldda,
            (rocblas_double_complex*) dB, lddb,
            (rocblas_double_complex*) &beta,
            (rocblas_double_complex*) dC, lddc ) );
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
        rocblas_sgemm_batched(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
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
        rocblas_dgemm_batched(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
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
        rocblas_cgemm_batched(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
            m, n, k,
            (rocblas_float_complex*)        &alpha,
            (rocblas_float_complex const**) dAarray, ldda,
            (rocblas_float_complex const**) dBarray, lddb,
            (rocblas_float_complex*)        &beta,
            (rocblas_float_complex**)       dCarray, lddc,
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
        rocblas_zgemm_batched(
            queue.handle(),
            op2rocblas(transA), op2rocblas(transB),
            m, n, k,
            (rocblas_double_complex*)        &alpha,
            (rocblas_double_complex const**) dAarray, ldda,
            (rocblas_double_complex const**) dBarray, lddb,
            (rocblas_double_complex*)        &beta,
            (rocblas_double_complex**)       dCarray, lddc,
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
        rocblas_strsm_batched(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
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
        rocblas_dtrsm_batched(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
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
        rocblas_ctrsm_batched(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_float_complex*)        &alpha,
            (rocblas_float_complex const**) dAarray, ldda,
            (rocblas_float_complex**)       dBarray, lddb,
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
        rocblas_ztrsm_batched(
            queue.handle(),
            side2rocblas(side), uplo2rocblas(uplo), op2rocblas(trans), diag2rocblas(diag),
            m, n,
            (rocblas_double_complex*)        &alpha,
            (rocblas_double_complex const**) dAarray, ldda,
            (rocblas_double_complex**)       dBarray, lddb,
            batch_size ) );
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS_HAVE_ROCBLAS
