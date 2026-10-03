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
    // OpenBLAS built with SYMBOLSUFFIX (e.g., 64_) also suffixes CBLAS
    // functions, exporting cblas_dgemm64_. Defined before including cblas.h,
    // these work whether or not the header's declarations are suffixed.
    #ifdef BLAS_FORTRAN_SUFFIX
        #define BLAS_CBLAS_CONCAT_( a, b ) a##b
        #define BLAS_CBLAS_CONCAT(  a, b ) BLAS_CBLAS_CONCAT_( a, b )
        #define BLAS_CBLAS_NAME( name ) \
            BLAS_CBLAS_CONCAT( name, BLAS_FORTRAN_SUFFIX )

        #define cblas_sasum      BLAS_CBLAS_NAME( cblas_sasum )
        #define cblas_dasum      BLAS_CBLAS_NAME( cblas_dasum )
        #define cblas_scasum     BLAS_CBLAS_NAME( cblas_scasum )
        #define cblas_dzasum     BLAS_CBLAS_NAME( cblas_dzasum )
        #define cblas_saxpy      BLAS_CBLAS_NAME( cblas_saxpy )
        #define cblas_daxpy      BLAS_CBLAS_NAME( cblas_daxpy )
        #define cblas_caxpy      BLAS_CBLAS_NAME( cblas_caxpy )
        #define cblas_zaxpy      BLAS_CBLAS_NAME( cblas_zaxpy )
        #define cblas_scopy      BLAS_CBLAS_NAME( cblas_scopy )
        #define cblas_dcopy      BLAS_CBLAS_NAME( cblas_dcopy )
        #define cblas_ccopy      BLAS_CBLAS_NAME( cblas_ccopy )
        #define cblas_zcopy      BLAS_CBLAS_NAME( cblas_zcopy )
        #define cblas_sdot       BLAS_CBLAS_NAME( cblas_sdot )
        #define cblas_ddot       BLAS_CBLAS_NAME( cblas_ddot )
        #define cblas_cdotc_sub  BLAS_CBLAS_NAME( cblas_cdotc_sub )
        #define cblas_zdotc_sub  BLAS_CBLAS_NAME( cblas_zdotc_sub )
        #define cblas_cdotu_sub  BLAS_CBLAS_NAME( cblas_cdotu_sub )
        #define cblas_zdotu_sub  BLAS_CBLAS_NAME( cblas_zdotu_sub )
        #define cblas_isamax     BLAS_CBLAS_NAME( cblas_isamax )
        #define cblas_idamax     BLAS_CBLAS_NAME( cblas_idamax )
        #define cblas_icamax     BLAS_CBLAS_NAME( cblas_icamax )
        #define cblas_izamax     BLAS_CBLAS_NAME( cblas_izamax )
        #define cblas_snrm2      BLAS_CBLAS_NAME( cblas_snrm2 )
        #define cblas_dnrm2      BLAS_CBLAS_NAME( cblas_dnrm2 )
        #define cblas_scnrm2     BLAS_CBLAS_NAME( cblas_scnrm2 )
        #define cblas_dznrm2     BLAS_CBLAS_NAME( cblas_dznrm2 )
        #define cblas_srot       BLAS_CBLAS_NAME( cblas_srot )
        #define cblas_drot       BLAS_CBLAS_NAME( cblas_drot )
        #define cblas_srotg      BLAS_CBLAS_NAME( cblas_srotg )
        #define cblas_drotg      BLAS_CBLAS_NAME( cblas_drotg )
        #define cblas_srotm      BLAS_CBLAS_NAME( cblas_srotm )
        #define cblas_drotm      BLAS_CBLAS_NAME( cblas_drotm )
        #define cblas_srotmg     BLAS_CBLAS_NAME( cblas_srotmg )
        #define cblas_drotmg     BLAS_CBLAS_NAME( cblas_drotmg )
        #define cblas_sscal      BLAS_CBLAS_NAME( cblas_sscal )
        #define cblas_dscal      BLAS_CBLAS_NAME( cblas_dscal )
        #define cblas_cscal      BLAS_CBLAS_NAME( cblas_cscal )
        #define cblas_zscal      BLAS_CBLAS_NAME( cblas_zscal )
        #define cblas_sswap      BLAS_CBLAS_NAME( cblas_sswap )
        #define cblas_dswap      BLAS_CBLAS_NAME( cblas_dswap )
        #define cblas_cswap      BLAS_CBLAS_NAME( cblas_cswap )
        #define cblas_zswap      BLAS_CBLAS_NAME( cblas_zswap )
        #define cblas_sgemv      BLAS_CBLAS_NAME( cblas_sgemv )
        #define cblas_dgemv      BLAS_CBLAS_NAME( cblas_dgemv )
        #define cblas_cgemv      BLAS_CBLAS_NAME( cblas_cgemv )
        #define cblas_zgemv      BLAS_CBLAS_NAME( cblas_zgemv )
        #define cblas_ssymv      BLAS_CBLAS_NAME( cblas_ssymv )
        #define cblas_dsymv      BLAS_CBLAS_NAME( cblas_dsymv )
        #define cblas_chemv      BLAS_CBLAS_NAME( cblas_chemv )
        #define cblas_zhemv      BLAS_CBLAS_NAME( cblas_zhemv )
        #define cblas_strmv      BLAS_CBLAS_NAME( cblas_strmv )
        #define cblas_dtrmv      BLAS_CBLAS_NAME( cblas_dtrmv )
        #define cblas_ctrmv      BLAS_CBLAS_NAME( cblas_ctrmv )
        #define cblas_ztrmv      BLAS_CBLAS_NAME( cblas_ztrmv )
        #define cblas_strsv      BLAS_CBLAS_NAME( cblas_strsv )
        #define cblas_dtrsv      BLAS_CBLAS_NAME( cblas_dtrsv )
        #define cblas_ctrsv      BLAS_CBLAS_NAME( cblas_ctrsv )
        #define cblas_ztrsv      BLAS_CBLAS_NAME( cblas_ztrsv )
        #define cblas_sger       BLAS_CBLAS_NAME( cblas_sger )
        #define cblas_dger       BLAS_CBLAS_NAME( cblas_dger )
        #define cblas_cgerc      BLAS_CBLAS_NAME( cblas_cgerc )
        #define cblas_zgerc      BLAS_CBLAS_NAME( cblas_zgerc )
        #define cblas_cgeru      BLAS_CBLAS_NAME( cblas_cgeru )
        #define cblas_zgeru      BLAS_CBLAS_NAME( cblas_zgeru )
        #define cblas_ssyr       BLAS_CBLAS_NAME( cblas_ssyr )
        #define cblas_dsyr       BLAS_CBLAS_NAME( cblas_dsyr )
        #define cblas_cher       BLAS_CBLAS_NAME( cblas_cher )
        #define cblas_zher       BLAS_CBLAS_NAME( cblas_zher )
        #define cblas_ssyr2      BLAS_CBLAS_NAME( cblas_ssyr2 )
        #define cblas_dsyr2      BLAS_CBLAS_NAME( cblas_dsyr2 )
        #define cblas_cher2      BLAS_CBLAS_NAME( cblas_cher2 )
        #define cblas_zher2      BLAS_CBLAS_NAME( cblas_zher2 )
        #define cblas_sgemm      BLAS_CBLAS_NAME( cblas_sgemm )
        #define cblas_dgemm      BLAS_CBLAS_NAME( cblas_dgemm )
        #define cblas_cgemm      BLAS_CBLAS_NAME( cblas_cgemm )
        #define cblas_zgemm      BLAS_CBLAS_NAME( cblas_zgemm )
        #define cblas_ssymm      BLAS_CBLAS_NAME( cblas_ssymm )
        #define cblas_dsymm      BLAS_CBLAS_NAME( cblas_dsymm )
        #define cblas_chemm      BLAS_CBLAS_NAME( cblas_chemm )
        #define cblas_zhemm      BLAS_CBLAS_NAME( cblas_zhemm )
        #define cblas_csymm      BLAS_CBLAS_NAME( cblas_csymm )
        #define cblas_zsymm      BLAS_CBLAS_NAME( cblas_zsymm )
        #define cblas_ssyrk      BLAS_CBLAS_NAME( cblas_ssyrk )
        #define cblas_dsyrk      BLAS_CBLAS_NAME( cblas_dsyrk )
        #define cblas_cherk      BLAS_CBLAS_NAME( cblas_cherk )
        #define cblas_zherk      BLAS_CBLAS_NAME( cblas_zherk )
        #define cblas_csyrk      BLAS_CBLAS_NAME( cblas_csyrk )
        #define cblas_zsyrk      BLAS_CBLAS_NAME( cblas_zsyrk )
        #define cblas_ssyr2k     BLAS_CBLAS_NAME( cblas_ssyr2k )
        #define cblas_dsyr2k     BLAS_CBLAS_NAME( cblas_dsyr2k )
        #define cblas_cher2k     BLAS_CBLAS_NAME( cblas_cher2k )
        #define cblas_zher2k     BLAS_CBLAS_NAME( cblas_zher2k )
        #define cblas_csyr2k     BLAS_CBLAS_NAME( cblas_csyr2k )
        #define cblas_zsyr2k     BLAS_CBLAS_NAME( cblas_zsyr2k )
        #define cblas_strmm      BLAS_CBLAS_NAME( cblas_strmm )
        #define cblas_dtrmm      BLAS_CBLAS_NAME( cblas_dtrmm )
        #define cblas_ctrmm      BLAS_CBLAS_NAME( cblas_ctrmm )
        #define cblas_ztrmm      BLAS_CBLAS_NAME( cblas_ztrmm )
        #define cblas_strsm      BLAS_CBLAS_NAME( cblas_strsm )
        #define cblas_dtrsm      BLAS_CBLAS_NAME( cblas_dtrsm )
        #define cblas_ctrsm      BLAS_CBLAS_NAME( cblas_ctrsm )
        #define cblas_ztrsm      BLAS_CBLAS_NAME( cblas_ztrsm )
    #endif

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
// (Fixed in LAPACK PR #721, 2022-10)
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
void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<float> alpha,
    std::complex<float> const* A, int lda,
    std::complex<float> const* x, int incx,
    std::complex<float> beta,
    std::complex<float>* yref, int incy );

void
cblas_symv(
    CBLAS_LAYOUT layout,
    CBLAS_UPLO uplo,
    int n,
    std::complex<double> alpha,
    std::complex<double> const* A, int lda,
    std::complex<double> const* x, int incx,
    std::complex<double> beta,
    std::complex<double>* yref, int incy );

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

// LAPACK provides [cz]syr, CBLAS lacks them
void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<float> alpha,
    std::complex<float> const *x, int incx,
    std::complex<float>* A, int lda );

void
cblas_syr(
    CBLAS_LAYOUT layout, CBLAS_UPLO uplo, int n,
    std::complex<double> alpha,
    std::complex<double> const *x, int incx,
    std::complex<double>* A, int lda );

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

// -----------------------------------------------------------------------------
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
