// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/fortran.h"
#include "blas.hh"
#include "blas_internal.hh"
#include "blas/counter.hh"

#include <limits>
#include <string.h>

namespace blas {

//==============================================================================
namespace internal {

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, float version.
/// @ingroup gemm_internal
inline void gemm(
    char transA, char transB,
    blas_int m, blas_int n, blas_int k,
    float alpha,
    float const* A, blas_int lda,
    float const* B, blas_int ldb,
    float beta,
    float*       C, blas_int ldc )
{
    BLAS_sgemm( &transA, &transB, &m, &n, &k,
                &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, double version.
/// @ingroup gemm_internal
inline void gemm(
    char transA, char transB,
    blas_int m, blas_int n, blas_int k,
    double alpha,
    double const* A, blas_int lda,
    double const* B, blas_int ldb,
    double beta,
    double*       C, blas_int ldc )
{
    BLAS_dgemm( &transA, &transB, &m, &n, &k,
                &alpha, A, &lda, B, &ldb, &beta, C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<float> version.
/// @ingroup gemm_internal
inline void gemm(
    char transA, char transB,
    blas_int m, blas_int n, blas_int k,
    std::complex<float> alpha,
    std::complex<float> const* A, blas_int lda,
    std::complex<float> const* B, blas_int ldb,
    std::complex<float> beta,
    std::complex<float>*       C, blas_int ldc )
{
    BLAS_cgemm( &transA, &transB, &m, &n, &k,
                (blas_complex_float*) &alpha,
                (blas_complex_float*) A, &lda,
                (blas_complex_float*) B, &ldb,
                (blas_complex_float*) &beta,
                (blas_complex_float*) C, &ldc );
}

//------------------------------------------------------------------------------
/// Low-level overload wrapper calls Fortran, complex<double> version.
/// @ingroup gemm_internal
inline void gemm(
    char transA, char transB,
    blas_int m, blas_int n, blas_int k,
    std::complex<double> alpha,
    std::complex<double> const* A, blas_int lda,
    std::complex<double> const* B, blas_int ldb,
    std::complex<double> beta,
    std::complex<double>*       C, blas_int ldc )
{
    BLAS_zgemm( &transA, &transB, &m, &n, &k,
                (blas_complex_double*) &alpha,
                (blas_complex_double*) A, &lda,
                (blas_complex_double*) B, &ldb,
                (blas_complex_double*) &beta,
                (blas_complex_double*) C, &ldc );
}

}  // namespace internal

//==============================================================================
namespace impl {

//------------------------------------------------------------------------------
/// Mid-level templated wrapper checks and converts arguments,
/// then calls low-level wrapper.
/// @ingroup gemm_internal
///
template <typename scalar_t>
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    scalar_t alpha,
    scalar_t const* A, int64_t lda,
    scalar_t const* B, int64_t ldb,
    scalar_t beta,
    scalar_t*       C, int64_t ldc )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( transA != Op::NoTrans &&
                   transA != Op::Trans &&
                   transA != Op::ConjTrans );
    blas_error_if( transB != Op::NoTrans &&
                   transB != Op::Trans &&
                   transB != Op::ConjTrans );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if (layout == Layout::ColMajor) {
        if (transA == Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB == Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( lda < m );
        else
            blas_error_if( lda < k );

        if (transB != Op::NoTrans)
            blas_error_if( ldb < k );
        else
            blas_error_if( ldb < n );

        blas_error_if( ldc < n );
    }

    #ifdef BLAS_HAVE_PAPI
        // PAPI instrumentation
        counter::gemm_type element;
        memset( &element, 0, sizeof( element ) );
        element = { transA, transB, m, n, k };
        counter::insert( element, counter::Id::gemm );
    #endif

    // convert arguments
    blas_int m_   = to_blas_int( m );
    blas_int n_   = to_blas_int( n );
    blas_int k_   = to_blas_int( k );
    blas_int lda_ = to_blas_int( lda );
    blas_int ldb_ = to_blas_int( ldb );
    blas_int ldc_ = to_blas_int( ldc );
    char transA_ = op2char( transA );
    char transB_ = op2char( transB );

    // call low-level wrapper
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        internal::gemm( transB_, transA_, n_, m_, k_,
                        alpha, B, ldb_, A, lda_, beta, C, ldc_ );
    }
    else {
        internal::gemm( transA_, transB_, m_, n_, k_,
                        alpha, A, lda_, B, ldb_, beta, C, ldc_ );
    }
}

}  // namespace impl

//==============================================================================
// High-level overloaded wrappers call mid-level templated wrapper.
// Overloading allows C++ more flexibility to convert scalars,
// and allows for a separate templated implementation.
// When calling a template, all the templated arguments (e.g., scalar_t)
// must match types exactly.

//------------------------------------------------------------------------------
/// CPU, float version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const* A, int64_t lda,
    float const* B, int64_t ldb,
    float beta,
    float*       C, int64_t ldc )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, double version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const* A, int64_t lda,
    double const* B, int64_t ldb,
    double beta,
    double*       C, int64_t ldc )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<float> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const* A, int64_t lda,
    std::complex<float> const* B, int64_t ldb,
    std::complex<float> beta,
    std::complex<float>*       C, int64_t ldc )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

//------------------------------------------------------------------------------
/// CPU, complex<double> version.
/// @ingroup gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const* A, int64_t lda,
    std::complex<double> const* B, int64_t ldb,
    std::complex<double> beta,
    std::complex<double>*       C, int64_t ldc )
{
    impl::gemm( layout, transA, transB, m, n, k,
                alpha, A, lda, B, ldb, beta, C, ldc );
}

}  // namespace blas

#ifdef BLAS_HAVE_PAPI

// Hook for papi_native_avail utility. No user code which links against this library should call
// this function because it has the same name in all SDE-enabled libraries. papi_native_avail
// uses dlopen and dlclose on each library so it only has one version of this symbol at a time.
extern "C"
papi_handle_t papi_sde_hook_list_events( papi_sde_fptr_struct_t* fptr_struct )
{
    papi_handle_t tmp_handle;
    tmp_handle = fptr_struct->init( "blas" );
    fptr_struct->create_counting_set( tmp_handle, "counter", NULL );
    return tmp_handle;
}

#endif
