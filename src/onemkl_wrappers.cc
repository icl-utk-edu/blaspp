// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#ifdef BLAS_HAVE_ONEMKL

// see https://software.intel.com/content/www/us/en/develop/articles/use-of-intel-mkl-data-types-in-cc-applications.html
#include "blas/config.h"
#define MKL_Complex8  blas_complex_float
#define MKL_Complex16 blas_complex_double
#include <oneapi/mkl.hpp>

namespace blas {
namespace device {

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
/// @return the corresponding device diag constant
oneapi::mkl::diag diag2onemkl(blas::Diag diag)
{
    switch (diag) {
        case Diag::Unit:    return oneapi::mkl::diag::U; break;
        case Diag::NonUnit: return oneapi::mkl::diag::N; break;
        default: throw blas::Error( "unknown diag" );
    }
}

// -----------------------------------------------------------------------------
/// @return the corresponding device uplo constant
oneapi::mkl::uplo uplo2onemkl(blas::Uplo uplo)
{
    switch (uplo) {
        case Uplo::Upper: return oneapi::mkl::uplo::U; break;
        case Uplo::Lower: return oneapi::mkl::uplo::L; break;
        default: throw blas::Error( "unknown uplo" );
    }
}

// -----------------------------------------------------------------------------
/// @return the corresponding device side constant
oneapi::mkl::side side2onemkl(blas::Side side)
{
    switch (side) {
        case Side::Left:  return oneapi::mkl::side::L;  break;
        case Side::Right: return oneapi::mkl::side::R; break;
        default: throw blas::Error( "unknown side" );
    }
}

// =============================================================================
// Level 1 BLAS - Device Interfaces

// -----------------------------------------------------------------------------
// nrm2
// -----------------------------------------------------------------------------
// snrm2
void snrm2(
    blas::Queue& queue,
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *result)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::nrm2(
            dev_queue,
            n,
            dx, incdx,
            result));
}

// -----------------------------------------------------------------------------
// dnrm2
void dnrm2(
    blas::Queue& queue,
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *result)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::nrm2(
            dev_queue,
            n,
            dx, incdx,
            result));
}

// -----------------------------------------------------------------------------
// cnrm2
void cnrm2(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    float *result)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::nrm2(
            dev_queue,
            n,
            dx, incdx,
            result));
}

// -----------------------------------------------------------------------------
// znrm2
void znrm2(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    double result)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::nrm2(
            dev_queue,
            n,
            dx, incdx,
            result));
}
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// axpy
// -----------------------------------------------------------------------------
// saxpy
void saxpy(
    blas::Queue& queue,
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::axpy(
            dev_queue,
            n,
            alpha,
            dx, incdx,
            dy, incdy));
}

// -----------------------------------------------------------------------------
// daxpy
void daxpy(
    blas::Queue& queue,
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::axpy(
            dev_queue,
            n,
            alpha,
            dx, incdx,
            dy, incdy));
}

// -----------------------------------------------------------------------------
// caxpy
void caxpy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::axpy(
            dev_queue,
            n,
            alpha,
            dx, incdx,
            dy, incdy));
}

// -----------------------------------------------------------------------------
// zaxpy
void zaxpy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::axpy(
            dev_queue,
            n,
            alpha,
            dx, incdx,
            dy, incdy));
}
// -----------------------------------------------------------------------------
// scal
// -----------------------------------------------------------------------------
// sscal
void sscal(
    blas::Queue& queue,
    device_blas_int n,
    float alpha,
    float *dx, device_blas_int incdx)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::scal(
            dev_queue,
            n,
            alpha,
            dx, incdx));
}

// -----------------------------------------------------------------------------
// dscal
void dscal(
    blas::Queue& queue,
    device_blas_int n,
    double alpha,
    double *dx, device_blas_int incdx)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::scal(
            dev_queue,
            n,
            alpha,
            dx, incdx));
}

// -----------------------------------------------------------------------------
// cscal
void cscal(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> *dx, device_blas_int incdx)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::scal(
            dev_queue,
            n,
            alpha,
            dx, incdx));
}

// -----------------------------------------------------------------------------
// zscal
void zscal(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::scal(
            dev_queue,
            n,
            alpha,
            dx,
            incdx));
}

// -----------------------------------------------------------------------------
// swap
// -----------------------------------------------------------------------------
// sswap
void sswap(
    blas::Queue& queue,
    device_blas_int n,
    float *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::swap(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}
// -----------------------------------------------------------------------------
// dswap
void dswap(
    blas::Queue& queue,
    device_blas_int n,
    double *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::swap(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}
// -----------------------------------------------------------------------------
// cswap
void cswap(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::swap(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}
// -----------------------------------------------------------------------------
// zswap
void zswap(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::swap(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}

// -----------------------------------------------------------------------------
// copy
// -----------------------------------------------------------------------------
// scopy
void scopy(
    blas::Queue& queue,
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::copy(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}

// -----------------------------------------------------------------------------
// dcopy
void dcopy(
    blas::Queue& queue,
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::copy(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}

// -----------------------------------------------------------------------------
// ccopy
void ccopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::copy(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}

// -----------------------------------------------------------------------------
// zcopy
void zcopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::copy(
            dev_queue,
            n,
            dx, incdx,
            dy, incdy) );
}

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
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float beta,
    float       *dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            dev_queue,
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dgemm
void dgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double beta,
    double       *dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            dev_queue,
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// cgemm
void cgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            dev_queue,
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zgemm
void zgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            dev_queue,
            op2onemkl( transA ), op2onemkl( transB ),
            m, n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// trsm
// -----------------------------------------------------------------------------
// strsm
void strsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// dtrsm
void dtrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ctrsm
void ctrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ztrsm
void ztrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// trmm
// -----------------------------------------------------------------------------
// strmm
void strmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const *dA, device_blas_int ldda,
    float       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// dtrmm
void dtrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const *dA, device_blas_int ldda,
    double       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ctrmm
void ctrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// ztrmm
void ztrmm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>       *dB, device_blas_int lddb)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            op2onemkl( trans ), diag2onemkl( diag ),
            m, n,
            alpha,
            dA, ldda,
            dB, lddb ) );
}

// -----------------------------------------------------------------------------
// hemm
// -----------------------------------------------------------------------------
// chemm
void chemm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::hemm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zhemm
void zhemm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::hemm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// symm
// -----------------------------------------------------------------------------
// ssymm
void ssymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::symm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsymm
void dsymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::symm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csymm
void csymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::symm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsymm
void zsymm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::symm(
            dev_queue,
            side2onemkl( side ), uplo2onemkl( uplo ),
            m, n,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// herk
// -----------------------------------------------------------------------------
// cherk
void cherk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::herk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zherk
void zherk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::herk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// syrk
// -----------------------------------------------------------------------------
// ssyrk
void ssyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float alpha,
    float const *dA, device_blas_int ldda,
    float  beta,
    float* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsyrk
void dsyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double alpha,
    double const *dA, device_blas_int ldda,
    double  beta,
    double* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csyrk
void csyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsyrk
void zsyrk(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// her2k
// -----------------------------------------------------------------------------
// cher2k
void cher2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    float  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::her2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zher2k
void zher2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    double  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::her2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// syr2k
// -----------------------------------------------------------------------------
// ssyr2k
void ssyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    float  alpha,
    float const *dA, device_blas_int ldda,
    float const *dB, device_blas_int lddb,
    float  beta,
    float* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// dsyr2k
void dsyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    double  alpha,
    double const *dA, device_blas_int ldda,
    double const *dB, device_blas_int lddb,
    double  beta,
    double* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
         oneapi::mkl::blas::syr2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// csyr2k
void csyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<float>  alpha,
    std::complex<float> const *dA, device_blas_int ldda,
    std::complex<float> const *dB, device_blas_int lddb,
    std::complex<float>  beta,
    std::complex<float>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// zsyr2k
void zsyr2k(
    blas::Queue& queue,
    blas::Uplo uplo, blas::Op trans,
    device_blas_int n, device_blas_int k,
    std::complex<double>  alpha,
    std::complex<double> const *dA, device_blas_int ldda,
    std::complex<double> const *dB, device_blas_int lddb,
    std::complex<double>  beta,
    std::complex<double>* dC, device_blas_int lddc)
{
    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            dev_queue,
            uplo2onemkl( uplo ), op2onemkl( trans ),
            n, k,
            alpha, dA, ldda,
                   dB, lddb,
            beta,  dC, lddc ) );
}

// -----------------------------------------------------------------------------
// batch gemm
// -----------------------------------------------------------------------------
// batch sgemm
void batch_sgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    float beta,
    float** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const float**)dAarray, &ldda,
            (const float**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch sgemm - group api
void batch_sgemm(
    blas::Queue& queue,
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    float *alpha,
    float const * const * dAarray, device_blas_int *ldda,
    float const * const * dBarray, device_blas_int *lddb,
    float *beta,
    float** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size)
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for(auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const float**)dAarray, ldda,
            (const float**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch dgemm
void batch_dgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    double beta,
    double** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const double**)dAarray, &ldda,
            (const double**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch dgemm - group api
void batch_dgemm(
    blas::Queue& queue,
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    double *alpha,
    double const * const * dAarray, device_blas_int *ldda,
    double const * const * dBarray, device_blas_int *lddb,
    double *beta,
    double** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size)
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for(auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const double**)dAarray, ldda,
            (const double**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch cgemm
void batch_cgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    std::complex<float> beta,
    std::complex<float>** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const std::complex<float>**)dAarray, &ldda,
            (const std::complex<float>**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch cgemm - group api
void batch_cgemm(
    blas::Queue& queue,
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    std::complex<float> *alpha,
    std::complex<float> const * const * dAarray, device_blas_int *ldda,
    std::complex<float> const * const * dBarray, device_blas_int *lddb,
    std::complex<float> *beta,
    std::complex<float>** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size)
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for(auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const std::complex<float>**)dAarray, ldda,
            (const std::complex<float>**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch zgemm
void batch_zgemm(
    blas::Queue& queue,
    blas::Op transA, blas::Op transB,
    device_blas_int m, device_blas_int n, device_blas_int k,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    std::complex<double> beta,
    std::complex<double>** dCarray, device_blas_int lddc,
    device_blas_int batch_size)
{
    oneapi::mkl::transpose transA_ = op2onemkl( transA );
    oneapi::mkl::transpose transB_ = op2onemkl( transB );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            (const std::complex<double>**)dAarray, &ldda,
            (const std::complex<double>**)dBarray, &lddb,
            &beta,
            dCarray, &lddc,
            1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch zgemm - group api
void batch_zgemm(
    blas::Queue& queue,
    blas::Op *transA, blas::Op *transB,
    device_blas_int *m, device_blas_int *n, device_blas_int *k,
    std::complex<double> *alpha,
    std::complex<double> const * const * dAarray, device_blas_int *ldda,
    std::complex<double> const * const * dBarray, device_blas_int *lddb,
    std::complex<double> *beta,
    std::complex<double>** dCarray, device_blas_int *lddc,
    device_blas_int group_count, device_blas_int *group_size)
{
    // todo: probably move transA_/transB_ to blas::Queue
    std::vector<oneapi::mkl::transpose> transA_(group_count);
    std::vector<oneapi::mkl::transpose> transB_(group_count);
    for(auto i = 0; i < group_count; ++i) {
        transA_[i] = op2onemkl( transA[i] );
        transB_[i] = op2onemkl( transB[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            dev_queue,
            transA_.data(), transB_.data(),
            m, n, k, alpha,
            (const std::complex<double>**)dAarray, ldda,
            (const std::complex<double>**)dBarray, lddb,
            beta,
            dCarray, lddc,
            group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch strsm
void batch_strsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    float alpha,
    float const * const * dAarray, device_blas_int ldda,
    float const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)

{
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        (const float**)dAarray, &ldda,
        (      float**)dBarray, &lddb,
        1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch strsm - group api
void batch_strsm(
    blas::Queue& queue,
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    float *alpha,
    float const * const * dAarray, device_blas_int *ldda,
    float const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size)

{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for(auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        side_.data(), uplo_.data(), trans_.data(), diag_.data(),
        m, n,
        alpha,
        (const float**)dAarray, ldda,
        (      float**)dBarray, lddb,
        group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch dtrsm
void batch_dtrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    double alpha,
    double const * const * dAarray, device_blas_int ldda,
    double const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        (const double**)dAarray, &ldda,
        (      double**)dBarray, &lddb,
        1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch dtrsm - group api
void batch_dtrsm(
    blas::Queue& queue,
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    double *alpha,
    double const * const * dAarray, device_blas_int *ldda,
    double const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size)

{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for(auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        side_.data(), uplo_.data(), trans_.data(), diag_.data(),
        m, n,
        alpha,
        (const double**)dAarray, ldda,
        (      double**)dBarray, lddb,
        group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch ctrsm
void batch_ctrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<float> alpha,
    std::complex<float> const * const * dAarray, device_blas_int ldda,
    std::complex<float> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        (const std::complex<float>**)dAarray, &ldda,
        (      std::complex<float>**)dBarray, &lddb,
        1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch ctrsm - group api
void batch_ctrsm(
    blas::Queue& queue,
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    std::complex<float> *alpha,
    std::complex<float> const * const * dAarray, device_blas_int *ldda,
    std::complex<float> const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size)

{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for(auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        side_.data(), uplo_.data(), trans_.data(), diag_.data(),
        m, n,
        alpha,
        (const std::complex<float>**)dAarray, ldda,
        (      std::complex<float>**)dBarray, lddb,
        group_count, group_size ) );
}

// -----------------------------------------------------------------------------
// batch ztrsm
void batch_ztrsm(
    blas::Queue& queue,
    blas::Side side, blas::Uplo uplo, blas::Op trans, blas::Diag diag,
    device_blas_int m, device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> const * const * dAarray, device_blas_int ldda,
    std::complex<double> const * const * dBarray, device_blas_int lddb,
    device_blas_int batch_size)
{
    oneapi::mkl::side side_       = side2onemkl( side );
    oneapi::mkl::uplo uplo_       = uplo2onemkl( uplo );
    oneapi::mkl::transpose trans_ = op2onemkl( trans );
    oneapi::mkl::diag diag_       = diag2onemkl( diag );

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        (const std::complex<double>**)dAarray, &ldda,
        (      std::complex<double>**)dBarray, &lddb,
        1, &batch_size ) );
}

// -----------------------------------------------------------------------------
// batch ztrsm - group api
void batch_ztrsm(
    blas::Queue& queue,
    blas::Side *side, blas::Uplo *uplo, blas::Op *trans, blas::Diag *diag,
    device_blas_int *m, device_blas_int *n,
    std::complex<double> *alpha,
    std::complex<double> const * const * dAarray, device_blas_int *ldda,
    std::complex<double> const * const * dBarray, device_blas_int *lddb,
    device_blas_int group_count, device_blas_int *group_size)

{
    // todo: probably move options to blas::Queue
    std::vector<oneapi::mkl::side>      side_(group_count);
    std::vector<oneapi::mkl::uplo>      uplo_(group_count);
    std::vector<oneapi::mkl::transpose> trans_(group_count);
    std::vector<oneapi::mkl::diag>      diag_(group_count);

    for(auto i = 0; i < group_count; ++i) {
        side_[i]  = side2onemkl( side[i] );
        uplo_[i]  = uplo2onemkl( uplo[i] );
        trans_[i] = op2onemkl( trans[i] );
        diag_[i]  = diag2onemkl( diag[i] );
    }

    /// todo: This sync should not be here
    /// however, the routine sometimes fails if removed
    queue.sync();

    sycl::queue dev_queue = queue.stream();
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        dev_queue,
        side_.data(), uplo_.data(), trans_.data(), diag_.data(),
        m, n,
        alpha,
        (const std::complex<double>**)dAarray, ldda,
        (      std::complex<double>**)dBarray, lddb,
        group_count, group_size ) );
}

}  // namespace device
}  // namespace blas

#endif  // BLAS_HAVE_ONEMKL
