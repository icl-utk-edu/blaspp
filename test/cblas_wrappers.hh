// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef CBLAS_HH
#define CBLAS_HH

#include "blas/defines.h"

#if defined(BLAS_HAVE_MKL) || defined(BLAS_HAVE_SYCL)
    #if defined(BLAS_ILP64) && ! defined(MKL_ILP64)
        #define MKL_ILP64
    #endif
    #include <mkl_cblas.h>

#elif defined(BLAS_HAVE_ESSL)
    #if defined(BLAS_ILP64) && ! defined(_ESV6464)
        #define _ESV6464
    #endif
    #include <essl.h>

#elif defined(BLAS_HAVE_ACCELERATE)
    // On macOS, the official way to include cblas is via Accelerate.h.
    // Unfortunately with Xcode 10.3 and GNU g++ 9.3, that doesn't compile.
    // If we can find cblas.h, use it, otherwise use Accelerate.h.
    #ifdef BLAS_HAVE_ACCELERATE_CBLAS_H
        #include <cblas.h>
    #else
        #include <Accelerate/Accelerate.h>
    #endif
    typedef CBLAS_ORDER CBLAS_LAYOUT;

#else
    // Some ancient cblas.h don't include extern C. It's okay to nest.
    extern "C" {
    #include <cblas.h>
    }

    // Original cblas.h used CBLAS_ORDER; new uses CBLAS_LAYOUT and makes
    // CBLAS_ORDER a typedef. Make sure CBLAS_LAYOUT is defined.
    typedef CBLAS_ORDER CBLAS_LAYOUT;
#endif

#include "blas/util.hh"

#include <complex>

// =============================================================================
// constants

// -----------------------------------------------------------------------------
inline CBLAS_LAYOUT cblas_layout_const( blas::Layout layout )
{
    switch (layout) {
        case blas::Layout::RowMajor:  return CblasRowMajor;
        case blas::Layout::ColMajor:  return CblasColMajor;
        default: throw blas::Error();
    }
}

inline CBLAS_LAYOUT cblas_layout_const( char layout )
{
    switch (layout) {
        case 'r': case 'R': return CblasRowMajor;
        case 'c': case 'C': return CblasColMajor;
        default: throw blas::Error();
    }
}

inline char lapack_layout_const( CBLAS_LAYOUT layout )
{
    switch (layout) {
        case CblasRowMajor: return 'r';
        case CblasColMajor: return 'c';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_DIAG cblas_diag_const( blas::Diag diag )
{
    switch (diag) {
        case blas::Diag::NonUnit:  return CblasNonUnit;
        case blas::Diag::Unit:     return CblasUnit;
        default: throw blas::Error();
    }
}

inline CBLAS_DIAG cblas_diag_const( char diag )
{
    switch (diag) {
        case 'n': case 'N': return CblasNonUnit;
        case 'u': case 'U': return CblasUnit;
        default: throw blas::Error();
    }
}

inline char lapack_diag_const( CBLAS_DIAG diag )
{
    switch (diag) {
        case CblasNonUnit: return 'n';
        case CblasUnit: return 'u';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_SIDE cblas_side_const( blas::Side side )
{
    switch (side) {
        case blas::Side::Left:  return CblasLeft;
        case blas::Side::Right: return CblasRight;
        default: throw blas::Error();
    }
}

inline CBLAS_SIDE cblas_side_const( char side )
{
    switch (side) {
        case 'l': case 'L': return CblasLeft;
        case 'r': case 'R': return CblasRight;
        default: throw blas::Error();
    }
}

inline char lapack_side_const( CBLAS_SIDE side )
{
    switch (side) {
        case CblasLeft:  return 'l';
        case CblasRight: return 'r';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_TRANSPOSE cblas_trans_const( blas::Op trans )
{
    switch (trans) {
        case blas::Op::NoTrans:   return CblasNoTrans;
        case blas::Op::Trans:     return CblasTrans;
        case blas::Op::ConjTrans: return CblasConjTrans;
        default: throw blas::Error();
    }
}

inline CBLAS_TRANSPOSE cblas_trans_const( char trans )
{
    switch (trans) {
        case 'n': case 'N': return CblasNoTrans;
        case 't': case 'T': return CblasTrans;
        case 'c': case 'C': return CblasConjTrans;
        default: throw blas::Error();
    }
}

inline char lapack_trans_const( CBLAS_TRANSPOSE trans )
{
    switch (trans) {
        case CblasNoTrans:   return 'n';
        case CblasTrans:     return 't';
        case CblasConjTrans: return 'c';
        default: throw blas::Error();
    }
}


// -----------------------------------------------------------------------------
inline CBLAS_UPLO cblas_uplo_const( blas::Uplo uplo )
{
    switch (uplo) {
        case blas::Uplo::Lower: return CblasLower;
        case blas::Uplo::Upper: return CblasUpper;
        default: throw blas::Error();
    }
}

inline CBLAS_UPLO cblas_uplo_const( char uplo )
{
    switch (uplo) {
        case 'l': case 'L': return CblasLower;
        case 'u': case 'U': return CblasUpper;
        default: throw blas::Error();
    }
}

inline char lapack_uplo_const( CBLAS_UPLO uplo )
{
    switch (uplo) {
        case CblasLower: return 'l';
        case CblasUpper: return 'u';
        default: throw blas::Error();
    }
}


// =============================================================================
// Level 1 BLAS

// -----------------------------------------------------------------------------
inline float
cblas_asum(
    int n, float const *x, int incx )
{
    return cblas_sasum( n, x, incx );
}

inline double
cblas_asum(
    int n, double const *x, int incx )
{
    return cblas_dasum( n, x, incx );
}

inline float
cblas_asum(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scasum( n, x, incx );
}

inline double
cblas_asum(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dzasum( n, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_axpy(
    int n, float alpha,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_saxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, double alpha,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_daxpy( n, alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_caxpy( n, &alpha, x, incx, y, incy );
}

inline void
cblas_axpy(
    int n, std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zaxpy( n, &alpha, x, incx, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_copy(
    int n,
    float const *x, int incx,
    float*       y, int incy )
{
    cblas_scopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    double const *x, int incx,
    double*       y, int incy )
{
    cblas_dcopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float>*       y, int incy )
{
    cblas_ccopy( n, x, incx, y, incy );
}

inline void
cblas_copy(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double>*       y, int incy )
{
    cblas_zcopy( n, x, incx, y, incy );
}


// -----------------------------------------------------------------------------
inline float
cblas_dot(
    int n,
    float const *x, int incx,
    float const *y, int incy )
{
    return cblas_sdot( n, x, incx, y, incy );
}

inline double
cblas_dot(
    int n,
    double const *x, int incx,
    double const *y, int incy )
{
    return cblas_ddot( n, x, incx, y, incy );
}

inline std::complex<float>
cblas_dot(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotc_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dot(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotc_sub( n, x, incx, y, incy, &result );
    return result;
}


// -----------------------------------------------------------------------------
// real dotu is same as dot
inline float
cblas_dotu(
    int n,
    float const *x, int incx,
    float const *y, int incy )
{
    return cblas_sdot( n, x, incx, y, incy );
}

inline double
cblas_dotu(
    int n,
    double const *x, int incx,
    double const *y, int incy )
{
    return cblas_ddot( n, x, incx, y, incy );
}

inline std::complex<float>
cblas_dotu(
    int n,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy )
{
    std::complex<float> result;
    cblas_cdotu_sub( n, x, incx, y, incy, &result );
    return result;
}

inline std::complex<double>
cblas_dotu(
    int n,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy )
{
    std::complex<double> result;
    cblas_zdotu_sub( n, x, incx, y, incy, &result );
    return result;
}


// -----------------------------------------------------------------------------
inline int
cblas_iamax(
    int n, float const *x, int incx )
{
    return cblas_isamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, double const *x, int incx )
{
    return cblas_idamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_icamax( n, x, incx );
}

inline int
cblas_iamax(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_izamax( n, x, incx );
}


// -----------------------------------------------------------------------------
inline float
cblas_nrm2(
    int n, float const *x, int incx )
{
    return cblas_snrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, double const *x, int incx )
{
    return cblas_dnrm2( n, x, incx );
}

inline float
cblas_nrm2(
    int n, std::complex<float> const *x, int incx )
{
    return cblas_scnrm2( n, x, incx );
}

inline double
cblas_nrm2(
    int n, std::complex<double> const *x, int incx )
{
    return cblas_dznrm2( n, x, incx );
}

// -----------------------------------------------------------------------------
inline void
cblas_rot(
    int n,
    float *x, int incx,
    float *y, int incy,
    float c, float s )
{
    cblas_srot( n, x, incx, y, incy, c, s );
}

inline void
cblas_rot(
    int n,
    double *x, int incx,
    double *y, int incy,
    double c, double s )
{
    cblas_drot( n, x, incx, y, incy, c, s );
}

void
cblas_rot(
    int n,
    std::complex<float> *x, int incx,
    std::complex<float> *y, int incy,
    float c, std::complex<float> s );

void
cblas_rot(
    int n,
    std::complex<double> *x, int incx,
    std::complex<double> *y, int incy,
    double c, std::complex<double> s );

// -----------------------------------------------------------------------------
inline void
cblas_rotg(
    float *a, float *b,
    float *c, float *s )
{
    cblas_srotg( a, b, c, s );
}

inline void
cblas_rotg(
    double *a, double *b,
    double *c, double *s )
{
    cblas_drotg( a, b, c, s );
}

// CBLAS lacks [cz]rotg, but they're in Netlib BLAS.
// Note c is real.
void
cblas_rotg(
    std::complex<float> *a, std::complex<float> *b,
    float *c, std::complex<float> *s );

void
cblas_rotg(
    std::complex<double> *a, std::complex<double> *b,
    double *c, std::complex<double> *s );


// -----------------------------------------------------------------------------
inline void
cblas_rotm(
    int n,
    float *x, int incx,
    float *y, int incy,
    float  p[5] )
{
    cblas_srotm( n, x, incx, y, incy, p );
}

inline void
cblas_rotm(
    int n,
    double *x, int incx,
    double *y, int incy,
    double  p[5] )
{
    cblas_drotm( n, x, incx, y, incy, p );
}


// -----------------------------------------------------------------------------
inline void
cblas_rotmg(
    float *d1, float *d2, float *x1, float y1, float param[5] )
{
    cblas_srotmg( d1, d2, x1, y1, param );
}

inline void
cblas_rotmg(
    double *d1, double *d2, double *x1, double y1, double param[5] )
{
    cblas_drotmg( d1, d2, x1, y1, param );
}


// -----------------------------------------------------------------------------
inline void
cblas_scal(
    int n, float alpha,
    float* x, int incx )
{
    cblas_sscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, double alpha,
    double* x, int incx )
{
    cblas_dscal( n, alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<float> alpha,
    std::complex<float>* x, int incx )
{
    cblas_cscal( n, &alpha, x, incx );
}

inline void
cblas_scal(
    int n, std::complex<double> alpha,
    std::complex<double>* x, int incx )
{
    cblas_zscal( n, &alpha, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_swap(
    int n,
    float* x, int incx,
    float* y, int incy )
{
    cblas_sswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    double* x, int incx,
    double* y, int incy )
{
    cblas_dswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<float>* x, int incx,
    std::complex<float>* y, int incy )
{
    cblas_cswap( n, x, incx, y, incy );
}

inline void
cblas_swap(
    int n,
    std::complex<double>* x, int incx,
    std::complex<double>* y, int incy )
{
    cblas_zswap( n, x, incx, y, incy );
}


// =============================================================================
// Level 2 BLAS

// -----------------------------------------------------------------------------
inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_sgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dgemv( layout, trans, m, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_cgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_gemv(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zgemv( layout, trans, m, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *x, int incx,
    std::complex<float>  beta,
    std::complex<float>* y, int incy )
{
    cblas_chemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}

inline void
cblas_hemv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *x, int incx,
    std::complex<double>  beta,
    std::complex<double>* y, int incy )
{
    cblas_zhemv( layout, uplo, n,
                 &alpha, A, lda, x, incx,
                 &beta, y, incy );
}


// -----------------------------------------------------------------------------
inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float  alpha,
    float const *A, int lda,
    float const *x, int incx,
    float  beta,
    float* y, int incy )
{
    cblas_ssymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

inline void
cblas_symv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double  alpha,
    double const *A, int lda,
    double const *x, int incx,
    double  beta,
    double* y, int incy )
{
    cblas_dsymv( layout, uplo, n,
                 alpha, A, lda, x, incx, beta, y, incy );
}

// LAPACK provides [cz]symv, CBLAS lacks them


// -----------------------------------------------------------------------------
inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trmv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrmv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    float const *A, int lda,
    float* x, int incx )
{
    cblas_strsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    double const *A, int lda,
    double* x, int incx )
{
    cblas_dtrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<float> const *A, int lda,
    std::complex<float>* x, int incx )
{
    cblas_ctrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}

inline void
cblas_trsv(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, CBLAS_DIAG diag, int n,
    std::complex<double> const *A, int lda,
    std::complex<double>* x, int incx )
{
    cblas_ztrsv( layout, uplo, trans, diag, n,
                 A, lda, x, incx );
}


// -----------------------------------------------------------------------------
inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_ger(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_gerc(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgerc( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_sger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dger( layout, m, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}

inline void
cblas_geru(
    CBLAS_LAYOUT layout, int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zgeru( layout, m, n, &alpha,
                 x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda )
{
    cblas_cher( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_her(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda )
{
    cblas_zher( layout, uplo, n, alpha, x, incx, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float* A, int lda )
{
    cblas_ssyr( layout, uplo, n, alpha, x, incx, A, lda );
}

inline void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double* A, int lda )
{
    cblas_dsyr( layout, uplo, n, alpha, x, incx, A, lda );
}

// -----------------------------------------------------------------------------
inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_ssyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dsyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    cblas_cher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_her2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    cblas_zher2( layout, uplo, n, &alpha, x, incx, y, incy, A, lda );
}


// -----------------------------------------------------------------------------
inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    float alpha,
    float const *x, int incx,
    float const *y, int incy,
    float* A, int lda )
{
    cblas_ssyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    double alpha,
    double const *x, int incx,
    double const *y, int incy,
    double* A, int lda )
{
    cblas_dsyr2( layout, uplo, n, alpha, x, incx, y, incy, A, lda );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float> const *y, int incy,
    std::complex<float>* A, int lda )
{
    fprintf( stderr, "csyr2 unavailable\n" );
}

inline void
cblas_syr2(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double> const *y, int incy,
    std::complex<double>* A, int lda )
{
    fprintf( stderr, "zsyr2 unavailable\n" );
}


// =============================================================================
// Level 3 BLAS

#if defined(BLAS_HAVE_MKL)
// -----------------------------------------------------------------------------
inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    blas::float16  alpha,
    blas::float16 const *A, int lda,
    blas::float16 const *B, int ldb,
    blas::float16  beta,
    blas::float16* C, int ldc )
{
    cblas_hgemm( layout, transA, transB, m, n, k,
                 (MKL_F16)alpha, (MKL_F16*)A, lda, (MKL_F16*)B, ldb,
                 (MKL_F16)beta,  (MKL_F16*)C, ldc );
}
#endif

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_sgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dgemm( layout, transA, transB, m, n, k,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_cgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_gemm(
    CBLAS_LAYOUT layout, CBLAS_TRANSPOSE transA, CBLAS_TRANSPOSE transB,
    int m, int n, int k,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zgemm( layout, transA, transB, m, n, k,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_chemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_hemm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zhemm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

// -----------------------------------------------------------------------------
inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    float  alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float  beta,
    float* C, int ldc )
{
    cblas_ssymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    double  alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double  beta,
    double* C, int ldc )
{
    cblas_dsymm( layout, side, uplo, m, n,
                 alpha, A, lda, B, ldb,
                 beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<float>  alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float>  beta,
    std::complex<float>* C, int ldc )
{
    cblas_csymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}

inline void
cblas_symm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    int m, int n,
    std::complex<double>  alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double>  beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsymm( layout, side, uplo, m, n,
                 &alpha, A, lda, B, ldb,
                 &beta,  C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_herk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,  // note: real
    std::complex<float> const *A, int lda,
    float beta,   // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_herk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,  // note: real
    std::complex<double> const *A, int lda,
    double beta,   // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zherk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_syrk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyrk( layout, uplo, trans, n, k, alpha, A, lda, beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}

inline void
cblas_syrk(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyrk( layout, uplo, trans, n, k, &alpha, A, lda, &beta, C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    float beta,  // note: real
    std::complex<float>*       C, int ldc )
{
    cblas_cher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_her2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    double beta,  // note: real
    std::complex<double>* C, int ldc )
{
    cblas_zher2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, beta, C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    float alpha,
    float const *A, int lda,
    float const *B, int ldb,
    float beta,
    float*       C, int ldc )
{
    cblas_ssyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    double alpha,
    double const *A, int lda,
    double const *B, int ldb,
    double beta,
    double*       C, int ldc )
{
    cblas_dsyr2k( layout, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float> const *B, int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int ldc )
{
    cblas_csyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}

inline void
cblas_syr2k(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double> const *B, int ldb,
    std::complex<double> beta,
    std::complex<double>* C, int ldc )
{
    cblas_zsyr2k( layout, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc );
}


// -----------------------------------------------------------------------------
inline void
cblas_trmm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    float alpha,
    float const *A, int lda,
    float       *B, int ldb )
{
    cblas_strmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trmm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    double alpha,
    double const *A, int lda,
    double       *B, int ldb )
{
    cblas_dtrmm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trmm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float>       *B, int ldb )
{
    cblas_ctrmm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

inline void
cblas_trmm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double>       *B, int ldb )
{
    cblas_ztrmm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}


// -----------------------------------------------------------------------------
inline void
cblas_trsm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    float alpha,
    float const *A, int lda,
    float       *B, int ldb )
{
    cblas_strsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    double alpha,
    double const *A, int lda,
    double       *B, int ldb )
{
    cblas_dtrsm( layout, side, uplo, trans, diag, m, n,  alpha, A, lda, B, ldb);
}

inline void
cblas_trsm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<float> alpha,
    std::complex<float> const *A, int lda,
    std::complex<float>       *B, int ldb )
{
    cblas_ctrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

inline void
cblas_trsm(
    CBLAS_LAYOUT layout, CBLAS_SIDE side, CBLAS_UPLO uplo,
    CBLAS_TRANSPOSE trans, CBLAS_DIAG diag,
    int m, int n,
    std::complex<double> alpha,
    std::complex<double> const *A, int lda,
    std::complex<double>       *B, int ldb )
{
    cblas_ztrsm( layout, side, uplo, trans, diag, m, n, &alpha, A, lda, B, ldb );
}

#endif        //  #ifndef CBLAS_HH
