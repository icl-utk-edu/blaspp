// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#ifdef BLAS_HAVE_CUBLAS

namespace blas {
namespace device {

// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
/// @return the corresponding device diag constant
cublasDiagType_t diag2cublas(blas::Diag diag)
{
    switch (diag) {
        case Diag::Unit:    return CUBLAS_DIAG_UNIT;     break;
        case Diag::NonUnit: return CUBLAS_DIAG_NON_UNIT; break;
        default: throw blas::Error( "unknown diag" );
    }
}

// -----------------------------------------------------------------------------
/// @return the corresponding device uplo constant
cublasFillMode_t uplo2cublas(blas::Uplo uplo)
{
    switch (uplo) {
        case Uplo::Upper: return CUBLAS_FILL_MODE_UPPER; break;
        case Uplo::Lower: return CUBLAS_FILL_MODE_LOWER; break;
        default: throw blas::Error( "unknown uplo" );
    }
}

// -----------------------------------------------------------------------------
/// @return the corresponding device side constant
cublasSideMode_t side2cublas(blas::Side side)
{
    switch (side) {
        case Side::Left:  return CUBLAS_SIDE_LEFT;  break;
        case Side::Right: return CUBLAS_SIDE_RIGHT; break;
        default: throw blas::Error( "unknown side" );
    }
}

// =============================================================================
// Level 1 BLAS - Device Interfaces

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
    blas_dev_call(
        cublasSaxpy(
            queue.handle(),
            n, &alpha,
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
    blas_dev_call(
        cublasDaxpy(
            queue.handle(),
            n, &alpha,
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
    blas_dev_call(
        cublasCaxpy(
            queue.handle(),
            n, (const cuComplex*) &alpha,
            (const cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy));
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
    blas_dev_call(
        cublasZaxpy(
            queue.handle(),
            n, (const cuDoubleComplex*) &alpha,
            (const cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy));
}

// -----------------------------------------------------------------------------
// dot
// -----------------------------------------------------------------------------
// sdot
void sdot(
    blas::Queue& queue,
    device_blas_int n,
    float const *dx, device_blas_int incdx,
    float const *dy, device_blas_int incdy,
    float *result)
{
    blas_dev_call(
        cublasSdot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            result));
}

// -----------------------------------------------------------------------------
// ddot
void ddot(
    blas::Queue& queue,
    device_blas_int n,
    double const *dx, device_blas_int incdx,
    double const *dy, device_blas_int incdy,
    double *result)
{
    blas_dev_call(
        cublasDdot(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy,
            result));
}

// -----------------------------------------------------------------------------
// cdotu
void cdotu(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result)
{
    blas_dev_call(
        cublasCdotu(
            queue.handle(),
            n,
            (const cuComplex*) dx, incdx,
            (const cuComplex*) dy, incdy,
            (cuComplex*) result));
}

// -----------------------------------------------------------------------------
// zdotu
void zdotu(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result)
{
    blas_dev_call(
        cublasZdotu(
            queue.handle(),
            n,
            (const cuDoubleComplex*) dx, incdx,
            (const cuDoubleComplex*) dy, incdy,
            (cuDoubleComplex*) result));
}

// -----------------------------------------------------------------------------
// cdotc
void cdotc(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> const *dy, device_blas_int incdy,
    std::complex<float> *result)
{
    blas_dev_call(
        cublasCdotc(
            queue.handle(),
            n,
            (const cuComplex*) dx, incdx,
            (const cuComplex*) dy, incdy,
            (cuComplex*) result));
}

// -----------------------------------------------------------------------------
// zdotc
void zdotc(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> const *dy, device_blas_int incdy,
    std::complex<double> *result)
{
    blas_dev_call(
        cublasZdotc(
            queue.handle(),
            n,
            (const cuDoubleComplex*) dx, incdx,
            (const cuDoubleComplex*) dy, incdy,
            (cuDoubleComplex*) result));
}

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
    blas_dev_call(
        cublasSnrm2(
            queue.handle(),
            n, dx, incdx,
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
    blas_dev_call(
        cublasDnrm2(
            queue.handle(),
            n, dx, incdx,
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
    blas_dev_call(
        cublasScnrm2(
            queue.handle(),
            n, (const cuComplex*) dx, incdx,
            result));
}

// -----------------------------------------------------------------------------
// znrm2
void znrm2(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    double *result)
{
    blas_dev_call(
        cublasDznrm2(
            queue.handle(),
            n, (const cuDoubleComplex*) dx, incdx,
            result));
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
    blas_dev_call(
        cublasSscal(
            queue.handle(),
            n, &alpha,
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
    blas_dev_call(
        cublasDscal(
            queue.handle(),
            n, &alpha,
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
    blas_dev_call(
        cublasCscal(
            queue.handle(),
            n, (const cuComplex*) &alpha,
            (cuComplex*) dx, incdx));
}

// -----------------------------------------------------------------------------
// zscal
void zscal(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> alpha,
    std::complex<double> *dx, device_blas_int incdx)
{
    blas_dev_call(
        cublasZscal(
            queue.handle(),
            n, (const cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dx, incdx));
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
    blas_dev_call(
        cublasSswap(
            queue.handle(),
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
    blas_dev_call(
        cublasDswap(
            queue.handle(),
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
    blas_dev_call(
        cublasCswap(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy) );
}

// -----------------------------------------------------------------------------
// zswap
void zswap(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy)
{
    blas_dev_call(
        cublasZswap(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy) );
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
    blas_dev_call(
        cublasScopy(
            queue.handle(),
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
    blas_dev_call(
        cublasDcopy(
            queue.handle(),
            n,
            dx, incdx,
            dy, incdy) );
}

// -----------------------------------------------------------------------------
// cswap
void ccopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<float> const *dx, device_blas_int incdx,
    std::complex<float> *dy, device_blas_int incdy)
{
    blas_dev_call(
        cublasCcopy(
            queue.handle(),
            n,
            (cuComplex*) dx, incdx,
            (cuComplex*) dy, incdy) );
}

// -----------------------------------------------------------------------------
// zswap
void zcopy(
    blas::Queue& queue,
    device_blas_int n,
    std::complex<double> const *dx, device_blas_int incdx,
    std::complex<double> *dy, device_blas_int incdy)
{
    blas_dev_call(
        cublasZcopy(
            queue.handle(),
            n,
            (cuDoubleComplex*) dx, incdx,
            (cuDoubleComplex*) dy, incdy) );
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
    blas_dev_call(
        cublasSgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasDgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB),
            m, n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasCgemm(
            queue.handle(),
            op2cublas(transA), op2cublas(transB), m, n, k,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb,
            (cuComplex*) &beta,
            (cuComplex*) dC, lddc ) );
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
    blas_dev_call(
        cublasStrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
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
    blas_dev_call(
        cublasDtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            &alpha,
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
    blas_dev_call(
        cublasCtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuComplex*) &alpha,
            (cuComplex*) dA, ldda,
            (cuComplex*) dB, lddb ) );
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
    blas_dev_call(
        cublasZtrsm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo), op2cublas(trans), diag2cublas(diag),
            m, n,
            (cuDoubleComplex*) &alpha,
            (cuDoubleComplex*) dA, ldda,
            (cuDoubleComplex*) dB, lddb ) );
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
    blas_dev_call(
        cublasSsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasDsymm(
            queue.handle(),
            side2cublas(side), uplo2cublas(uplo),
            m, n,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasCherk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, (cuComplex*) dA, ldda,
            &beta,  (cuComplex*) dC, lddc ) );
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
    blas_dev_call(
        cublasZherk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, (cuDoubleComplex*) dA, ldda,
            &beta,  (cuDoubleComplex*) dC, lddc ) );
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
    blas_dev_call(
        cublasSsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasDsyrk(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasSsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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
    blas_dev_call(
        cublasDsyr2k(
            queue.handle(),
            uplo2cublas(uplo), op2cublas(trans),
            n, k,
            &alpha, dA, ldda,
                    dB, lddb,
            &beta,  dC, lddc ) );
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

}  // namespace device
}  // namespace blas

#endif  // BLAS_HAVE_CUBLAS
