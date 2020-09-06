// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device.hh"

#include <vector>

/** device blas++ **/

namespace blas {

// =============================================================================
// Level 1 BLAS

// =============================================================================
// Level 2 BLAS

// =============================================================================
// Level 3 BLAS
// -----------------------------------------------------------------------------
// gemm
void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void gemm(
    blas::Layout layout,
    blas::Op transA,
    blas::Op transB,
    int64_t m, int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// trsm
void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *dA, int64_t ldda,
    float       *dB, int64_t lddb,
    blas::Queue &queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *dA, int64_t ldda,
    double       *dB, int64_t lddb,
    blas::Queue  &queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float>       *dB, int64_t lddb,
    blas::Queue  &queue );

void trsm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double>       *dB, int64_t lddb,
    blas::Queue  &queue );

// -----------------------------------------------------------------------------
// trmm
void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    float alpha,
    float const *dA, int64_t ldda,
    float       *dB, int64_t lddb,
    blas::Queue &queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    double alpha,
    double const *dA, int64_t ldda,
    double       *dB, int64_t lddb,
    blas::Queue &queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float>       *dB, int64_t lddb,
    blas::Queue &queue );

void trmm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t m,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double>       *dB, int64_t lddb,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// hemm
void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// symm
void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void symm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// herk
void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,  // note: real
    std::complex<float> const *dA, int64_t ldda,
    float beta,   // note: real
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void herk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    std::complex<double> const *dA, int64_t ldda,
    double beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// syrk
void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// her2k
void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,  // note: complex
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    float beta,   // note: real
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void her2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,  // note: complex
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    double beta,  // note: real
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// syr2k
void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue );

void syr2k(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue );

namespace batch {

// =============================================================================
// Level 1 Batch BLAS

// =============================================================================
// Level 2 Batch BLAS

// =============================================================================
// Level 3 Batch BLAS
// -----------------------------------------------------------------------------
// batch gemm
void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<float >   const &alpha,
    std::vector<float*>   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>   const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >   const &beta,
    std::vector<float*>   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<double >  const &alpha,
    std::vector<double*>  const &Aarray, std::vector<int64_t>  const &ldda,
    std::vector<double*>  const &Barray, std::vector<int64_t>  const &lddb,
    std::vector<double >  const &beta,
    std::vector<double*>  const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                  std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<float>  >   const &alpha,
    std::vector< std::complex<float>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<float>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<float>  >   const &beta,
    std::vector< std::complex<float>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                  std::vector<int64_t>  &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector< std::complex<double>  >   const &alpha,
    std::vector< std::complex<double>* >   const &Aarray, std::vector<int64_t> const &ldda,
    std::vector< std::complex<double>* >   const &Barray, std::vector<int64_t> const &lddb,
    std::vector< std::complex<double>  >   const &beta,
    std::vector< std::complex<double>* >   const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                                   std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<float >   const &alpha,
    std::vector<float*>   const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<float*>   const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<float >   const &beta,
    std::vector<float*>   const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                 layout,
    std::vector<blas::Op> const &transA,
    std::vector<blas::Op> const &transB,
    std::vector<int64_t>  const &m,
    std::vector<int64_t>  const &n,
    std::vector<int64_t>  const &k,
    std::vector<double >  const &alpha,
    std::vector<double*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<double*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<double >  const &beta,
    std::vector<double*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                              layout,
    std::vector<blas::Op>              const &transA,
    std::vector<blas::Op>              const &transB,
    std::vector<int64_t>               const &m,
    std::vector<int64_t>               const &n,
    std::vector<int64_t>               const &k,
    std::vector<std::complex<float> >  const &alpha,
    std::vector<std::complex<float>*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >  const &beta,
    std::vector<std::complex<float>*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue );

void gemm(
    blas::Layout                              layout,
    std::vector<blas::Op>              const &transA,
    std::vector<blas::Op>              const &transB,
    std::vector<int64_t>               const &m,
    std::vector<int64_t>               const &n,
    std::vector<int64_t>               const &k,
    std::vector<std::complex<double> >  const &alpha,
    std::vector<std::complex<double>*>  const &Aarray,     std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>  const &Barray,     std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >  const &beta,
    std::vector<std::complex<double>*>  const &Carray,     std::vector<int64_t> const &lddc,
    std::vector<size_t>   const &group_size, std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch trsm
void trsm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void trsm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void trsm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void trsm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch trmm
void trmm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void trmm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void trmm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void trmm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch hemm
void hemm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void hemm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void hemm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void hemm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch symm
void symm(
    blas::Layout                  layout,
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m,
    std::vector<int64_t>    const &n,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void symm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void symm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void symm(
    blas::Layout                   layout,
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m,
    std::vector<int64_t>     const &n,
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch herk
void herk(
    blas::Layout                  layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void herk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void herk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<float>       const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >      const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

void herk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double>      const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch syrk
void syrk(
    blas::Layout                  layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void syrk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void syrk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float> > const &alpha,
    std::vector<std::complex<float>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float> > const &beta,
    std::vector<std::complex<float>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

void syrk(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double> > const &alpha,
    std::vector<std::complex<double>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double> > const &beta,
    std::vector<std::complex<double>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch her2k
void her2k(
    blas::Layout                  layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void her2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void her2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float>>      const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >                   const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

void her2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double>>      const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >                   const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

// -----------------------------------------------------------------------------
// batch syr2k
void syr2k(
    blas::Layout                  layout,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<int64_t>    const &n,
    std::vector<int64_t>    const &k,
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info,
    blas::Queue &queue );

void syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                     std::vector<int64_t>       &info,
    blas::Queue &queue );

void syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<float> > const &alpha,
    std::vector<std::complex<float>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> > const &beta,
    std::vector<std::complex<float>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

void syr2k(
    blas::Layout                   layout,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<blas::Op>    const &trans,
    std::vector<int64_t>     const &n,
    std::vector<int64_t>     const &k,
    std::vector<std::complex<double> > const &alpha,
    std::vector<std::complex<double>*> const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*> const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> > const &beta,
    std::vector<std::complex<double>*> const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch, std::vector<int64_t>       &info,
    blas::Queue &queue );

}  // namespace batch
}  // namespace blas
