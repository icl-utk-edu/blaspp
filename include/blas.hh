// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

/// @file
/// @mainpage BLAS++: C++ API for BLAS (Basic Linear Algebra Subroutines)
///
/// BLAS++ provides a modern C++ interface to the Basic Linear Algebra Subroutines,
/// supporting both CPU and GPU execution. This main header includes all BLAS operations.
///
/// # Features
///
/// - **Type-generic templates**: Single API for float, double, complex<float>, complex<double>
/// - **Multiple backends**: Reference C++, vendor BLAS (MKL, OpenBLAS), GPU (cuBLAS, rocBLAS, SYCL)
/// - **Modern C++**: Uses C++11/14 features, std::complex, strong typing
/// - **Performance counters**: Optional PAPI integration via counter.hh
/// - **Device support**: Asynchronous GPU operations via device_blas.hh
///
/// # Organization
///
/// Operations are organized by BLAS level:
///
/// - **Level 1**: Vector-vector operations (axpy, dot, nrm2, scal, etc.)
/// - **Level 2**: Matrix-vector operations (gemv, ger, trmv, etc.)
/// - **Level 3**: Matrix-matrix operations (gemm, trmm, herk, etc.)
///
/// # Basic Usage
///
/// @code
/// #include <blas.hh>
/// 
/// // Matrix-matrix multiply: C = alpha*A*B + beta*C
/// blas::gemm(blas::Layout::ColMajor,
///            blas::Op::NoTrans, blas::Op::NoTrans,
///            m, n, k,
///            alpha, A, lda,
///                   B, ldb,
///            beta,  C, ldc);
/// @endcode
///
/// # Device (GPU) Usage
///
/// @code
/// blas::Queue queue(device_id);
/// blas::gemm(blas::Layout::ColMajor,
///            blas::Op::NoTrans, blas::Op::NoTrans,
///            m, n, k,
///            alpha, d_A, lda,
///                   d_B, ldb,
///            beta,  d_C, ldc,
///            queue);
/// queue.sync();  // Wait for completion
/// @endcode
///
/// @see util.hh for enumerations (Layout, Op, Uplo, Diag, Side)
/// @see device.hh for Queue and device management
/// @see counter.hh for performance counting with PAPI
/// @see flops.hh for FLOP and bandwidth calculations

#ifndef BLAS_HH
#define BLAS_HH

#include "blas/defines.h"

#include "blas/counter.hh"

// Version is updated by make_release.py; DO NOT EDIT.
// Version 2025.05.28
#define BLASPP_VERSION 20250528

/// @namespace blas
/// BLAS (Basic Linear Algebra Subroutines)
namespace blas {

int blaspp_version();
const char* blaspp_id();

}  // namespace blas

#include "blas/wrappers.hh"

// =============================================================================
// Level 1 BLAS template implementations

#include "blas/asum.hh"
#include "blas/axpy.hh"
#include "blas/copy.hh"
#include "blas/dot.hh"
#include "blas/dotu.hh"
#include "blas/iamax.hh"
#include "blas/nrm2.hh"
#include "blas/rot.hh"
#include "blas/rotg.hh"
#include "blas/rotm.hh"
#include "blas/rotmg.hh"
#include "blas/scal.hh"
#include "blas/swap.hh"

// =============================================================================
// Level 2 BLAS template implementations

#include "blas/gemv.hh"
#include "blas/ger.hh"
#include "blas/geru.hh"
#include "blas/hemv.hh"
#include "blas/her.hh"
#include "blas/her2.hh"
#include "blas/symv.hh"
#include "blas/syr.hh"
#include "blas/syr2.hh"
#include "blas/trmv.hh"
#include "blas/trsv.hh"

// =============================================================================
// Level 3 BLAS template implementations

#include "blas/gemm.hh"
#include "blas/hemm.hh"
#include "blas/herk.hh"
#include "blas/her2k.hh"
#include "blas/symm.hh"
#include "blas/syrk.hh"
#include "blas/syr2k.hh"
#include "blas/trmm.hh"
#include "blas/trsm.hh"

// =============================================================================
// Device BLAS

#include "blas/device_blas.hh"

#endif        //  #ifndef BLAS_HH
