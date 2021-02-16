// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#ifdef BLAS_HAVE_ONEMKL

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
        case Diag::Unit:    return oneapi::mkl::diag::N; break;
        case Diag::NonUnit: return oneapi::mkl::diag::U; break;
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
        oneapi::mkl::blas::swap(
        queue.stream(),
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
        oneapi::mkl::blas::swap(
        queue.stream(),
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
        oneapi::mkl::blas::swap(
        queue.stream(),
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
    blas_dev_call(
        oneapi::mkl::blas::swap(
        queue.stream(),
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
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl(transA), op2onemkl(transB),
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
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl(transA), op2onemkl(transB),
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
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl(transA), op2onemkl(transB), m, n, k,
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
    blas_dev_call(
        oneapi::mkl::blas::gemm(
            queue.stream(),
            op2onemkl(transA), op2onemkl(transB),
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
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trsm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::trmm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo), op2onemkl(trans), diag2onemkl(diag),
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
    blas_dev_call(
        oneapi::mkl::blas::hemm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::hemm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::symm(
            queue.stream(),
            side2onemkl(side), uplo2onemkl(uplo),
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
    blas_dev_call(
        oneapi::mkl::blas::herk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::herk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syrk(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::her2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::her2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
         oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    blas_dev_call(
        oneapi::mkl::blas::syr2k(
            queue.stream(),
            uplo2onemkl(uplo), op2onemkl(trans),
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
    oneapi::mkl::transpose transA_ = op2onemkl(transA);
    oneapi::mkl::transpose transB_ = op2onemkl(transB);
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            dA_array, &ldda,
            dB_array, &lddb,
            &beta,
            dC_array, &lddc,
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
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA, transB,
            m, n, k, alpha,
            dA_array, ldda,
            dB_array, lddb,
            beta,
            dC_array, lddc,
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
    oneapi::mkl::transpose transA_ = op2onemkl(transA);
    oneapi::mkl::transpose transB_ = op2onemkl(transB);
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            dA_array, &ldda,
            dB_array, &lddb,
            &beta,
            dC_array, &lddc,
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
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA, transB,
            m, n, k, alpha,
            dA_array, ldda,
            dB_array, lddb,
            beta,
            dC_array, lddc,
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
    oneapi::mkl::transpose transA_ = op2onemkl(transA);
    oneapi::mkl::transpose transB_ = op2onemkl(transB);
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            dA_array, &ldda,
            dB_array, &lddb,
            &beta,
            dC_array, &lddc,
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
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA, transB,
            m, n, k, alpha,
            dA_array, ldda,
            dB_array, lddb,
            beta,
            dC_array, lddc,
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
    oneapi::mkl::transpose transA_ = op2onemkl(transA);
    oneapi::mkl::transpose transB_ = op2onemkl(transB);
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            &transA_, &transB_,
            &m, &n, &k, &alpha,
            dA_array, &ldda,
            dB_array, &lddb,
            &beta,
            dC_array, &lddc,
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
    blas_dev_call(
        oneapi::mkl::blas::gemm_batch(
            queue.stream(),
            transA, transB,
            m, n, k, alpha,
            dA_array, ldda,
            dB_array, lddb,
            beta,
            dC_array, lddc,
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
    oneapi::mkl::transpose side_  = op2onemkl(side);
    oneapi::mkl::transpose uplo_  = op2onemkl(uplo);
    oneapi::mkl::transpose trans_ = op2onemkl(trans);
    oneapi::mkl::transpose diag_  = op2onemkl(diag);
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        dA_array, &ldda,
        dB_array, &lddb,
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
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        side, uplo, trans, diag,
        m, n,
        alpha,
        dA_array, ldda,
        dB_array, lddb,
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
    oneapi::mkl::transpose side_  = op2onemkl(side);
    oneapi::mkl::transpose uplo_  = op2onemkl(uplo);
    oneapi::mkl::transpose trans_ = op2onemkl(trans);
    oneapi::mkl::transpose diag_  = op2onemkl(diag);
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        dA_array, &ldda,
        dB_array, &lddb,
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
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        side, uplo, trans, diag,
        m, n,
        alpha,
        dA_array, ldda,
        dB_array, lddb,
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
    oneapi::mkl::transpose side_  = op2onemkl(side);
    oneapi::mkl::transpose uplo_  = op2onemkl(uplo);
    oneapi::mkl::transpose trans_ = op2onemkl(trans);
    oneapi::mkl::transpose diag_  = op2onemkl(diag);
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        dA_array, &ldda,
        dB_array, &lddb,
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
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        side, uplo, trans, diag,
        m, n,
        alpha,
        dA_array, ldda,
        dB_array, lddb,
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
    oneapi::mkl::transpose side_  = op2onemkl(side);
    oneapi::mkl::transpose uplo_  = op2onemkl(uplo);
    oneapi::mkl::transpose trans_ = op2onemkl(trans);
    oneapi::mkl::transpose diag_  = op2onemkl(diag);
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        &side_, &uplo_ , &trans_, &diag_,
        &m, &n,
        &alpha,
        dA_array, &ldda,
        dB_array, &lddb,
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
    blas_dev_call(
        oneapi::mkl::blas::trsm_batch(
        queue.stream(),
        side, uplo, trans, diag,
        m, n,
        alpha,
        dA_array, ldda,
        dB_array, lddb,
        group_count, group_size ) );
}

}  // namespace device
}  // namespace blas

#endif  // BLAS_HAVE_ONEMKL
