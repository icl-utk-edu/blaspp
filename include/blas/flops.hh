// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_FLOPS_HH
#define BLAS_FLOPS_HH

#include "blas.hh"

namespace blas {

//==============================================================================
/// @defgroup flops Floating Point Operation Counting
/// @brief Functions and classes for counting floating-point operations (FLOPs)
///        and data transfer (bytes) in BLAS operations.
///
/// These utilities provide both:
/// - Gflop: Giga-floating-point operations counting
/// - Gbyte: Gigabyte data transfer counting
///
/// Useful for performance analysis and roofline modeling.
//==============================================================================

// =============================================================================
// Level 1 BLAS - Internal helper functions

// -----------------------------------------------------------------------------
/// @brief Multiplication count for asum operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (0 for asum)
/// @ingroup flops
inline double fmuls_asum( double n )
    { return 0; }

/// @brief Addition count for asum operation.
/// @param[in] n Vector length
/// @return Number of add operations (n-1)
/// @ingroup flops
inline double fadds_asum( double n )
    { return n-1; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for axpy operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (n)
/// @ingroup flops
inline double fmuls_axpy( double n )
    { return n; }

/// @brief Addition count for axpy operation.
/// @param[in] n Vector length
/// @return Number of add operations (n)
/// @ingroup flops
inline double fadds_axpy( double n )
    { return n; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for iamax operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (0 for iamax)
/// @ingroup flops
inline double fmuls_iamax( double n )
    { return 0; }

/// @brief Addition count for iamax operation.
/// n-1 compares, which are essentially adds (x > y is x - y > 0)
/// @param[in] n Vector length
/// @return Number of add operations (n-1)
/// @ingroup flops
inline double fadds_iamax( double n )
    { return n-1; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for nrm2 (Euclidean norm) operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (n for x²)
/// @ingroup flops
inline double fmuls_nrm2( double n )
    { return n; }

/// @brief Addition count for nrm2 operation.
/// @param[in] n Vector length
/// @return Number of add operations (n-1 for sum)
/// @ingroup flops
inline double fadds_nrm2( double n )
    { return n-1; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for dot product operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (n)
/// @ingroup flops
inline double fmuls_dot( double n )
    { return n; }

/// @brief Addition count for dot product operation.
/// @param[in] n Vector length  
/// @return Number of add operations (n-1)
/// @ingroup flops
inline double fadds_dot( double n )
    { return n-1; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for scal (scale vector) operation.
/// @param[in] n Vector length
/// @return Number of multiply operations (n)
/// @ingroup flops
inline double fmuls_scal( double n )
    { return n; }

/// @brief Addition count for scal operation.
/// @param[in] n Vector length
/// @return Number of add operations (0)
/// @ingroup flops
inline double fadds_scal( double n )
    { return 0; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for Givens rotation.
/// @param[in] n Vector length
/// @return Number of multiply operations (4n)
/// @ingroup flops
inline double fmuls_rot( double n )
    { return 4 * n; }

/// @brief Addition count for Givens rotation.
/// @param[in] n Vector length
/// @return Number of add operations (2n)
/// @ingroup flops
inline double fadds_rot( double n )
    { return 2 * n; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for modified Givens rotation.
/// @param[in] n Vector length
/// @return Number of multiply operations (2n)
/// @ingroup flops
inline double fmuls_rotm( double n )
    { return 2 * n; }

/// @brief Addition count for modified Givens rotation.
/// @param[in] n Vector length
/// @return Number of add operations (2n)
/// @ingroup flops
inline double fadds_rotm( double n )
    { return 2 * n; }

// =============================================================================
// Level 2 BLAS - Internal helper functions
// Most formulas assume alpha=1, beta=0 or 1; otherwise add lower-order terms.
// i.e., this is minimum flops and bandwidth that could be consumed.

// -----------------------------------------------------------------------------
/// @brief Multiplication count for general matrix-vector multiply.
/// @param[in] m Number of rows
/// @param[in] n Number of columns
/// @return Number of multiply operations (m*n)
/// @ingroup flops
inline double fmuls_gemv( double m, double n )
    { return m*n; }

/// @brief Addition count for general matrix-vector multiply.
/// @param[in] m Number of rows
/// @param[in] n Number of columns
/// @return Number of add operations (m*n)
/// @ingroup flops
inline double fadds_gemv( double m, double n )
    { return m*n; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for triangular matrix-vector multiply.
/// @param[in] n Matrix dimension
/// @return Number of multiply operations (n*(n+1)/2)
/// @ingroup flops
inline double fmuls_trmv( double n )
    { return 0.5*n*(n + 1); }

/// @brief Addition count for triangular matrix-vector multiply.
/// @param[in] n Matrix dimension
/// @return Number of add operations (n*(n-1)/2)
/// @ingroup flops
inline double fadds_trmv( double n )
    { return 0.5*n*(n - 1); }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for rank-1 update (ger).
/// @param[in] m Number of rows
/// @param[in] n Number of columns
/// @return Number of multiply operations (m*n)
/// @ingroup flops
inline double fmuls_ger( double m, double n )
    { return m*n; }

/// @brief Addition count for rank-1 update (ger).
/// @param[in] m Number of rows
/// @param[in] n Number of columns
/// @return Number of add operations (m*n)
/// @ingroup flops
inline double fadds_ger( double m, double n )
    { return m*n; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for general matrix-matrix multiply.
/// @param[in] m Number of rows of C
/// @param[in] n Number of columns of C
/// @param[in] k Inner dimension
/// @return Number of multiply operations (m*n*k)
/// @ingroup flops
inline double fmuls_gemm( double m, double n, double k )
    { return m*n*k; }

/// @brief Addition count for general matrix-matrix multiply.
/// @param[in] m Number of rows of C
/// @param[in] n Number of columns of C
/// @param[in] k Inner dimension
/// @return Number of add operations (m*n*k)
/// @ingroup flops
inline double fadds_gemm( double m, double n, double k )
    { return m*n*k; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for banded general matrix-matrix multiply.
///
/// Assumes band matrix A (m-by-k) with lower bandwidth kl and upper bandwidth ku,
/// and general matrix B (k-by-n). Accounts for triangular and trapezoidal cases
/// when the band extends beyond matrix boundaries.
///
/// @param[in] m Number of rows of A and C
/// @param[in] n Number of columns of B and C
/// @param[in] k Number of columns of A, rows of B
/// @param[in] kl Lower bandwidth of A
/// @param[in] ku Upper bandwidth of A
/// @return Number of multiply operations
/// @ingroup flops
inline double fmuls_gbmm( double m, double n, double k, double kl, double ku )
{
    if (m-kl > k)
        return (kl*k + (k+1)*k/2. - (k-ku-1)*(k-ku)/2.)*n;
    if (k-ku > m)
        return (ku*m - (m-kl-1)*(m-kl)/2. + (m+1)*m/2.)*n;
    return (m*k - (m-kl-1)*(m-kl)/2. - (k-ku-1)*(k-ku)/2.)*n;
}

/// @brief Addition count for banded general matrix-matrix multiply.
/// Assuming alpha=1, beta=1, adds are same as muls.
/// @param[in] m Number of rows of A and C
/// @param[in] n Number of columns of B and C
/// @param[in] k Number of columns of A, rows of B
/// @param[in] kl Lower bandwidth of A
/// @param[in] ku Upper bandwidth of A
/// @return Number of add operations
/// @ingroup flops
inline double fadds_gbmm( double m, double n, double k, double kl, double ku )
{
    return fmuls_gbmm( m, n, k, kl, ku );
}

// -----------------------------------------------------------------------------
/// @brief Multiplication count for Hermitian matrix-matrix multiply.
/// @param[in] side Side where Hermitian matrix appears (Left or Right)
/// @param[in] m Number of rows of C
/// @param[in] n Number of columns of C
/// @return Number of multiply operations
/// @ingroup flops
inline double fmuls_hemm( blas::Side side, double m, double n )
    { return (side == blas::Side::Left ? m*m*n : m*n*n); }

/// @brief Addition count for Hermitian matrix-matrix multiply.
/// @param[in] side Side where Hermitian matrix appears
/// @param[in] m Number of rows of C
/// @param[in] n Number of columns of C
/// @return Number of add operations
/// @ingroup flops
inline double fadds_hemm( blas::Side side, double m, double n )
    { return (side == blas::Side::Left ? m*m*n : m*n*n); }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for Hermitian rank-k update.
/// @param[in] n Dimension of Hermitian matrix C
/// @param[in] k Inner dimension
/// @return Number of multiply operations (k*n*(n+1)/2)
/// @ingroup flops
inline double fmuls_herk( double n, double k )
    { return 0.5*k*n*(n+1); }

/// @brief Addition count for Hermitian rank-k update.
/// @param[in] n Dimension of Hermitian matrix C
/// @param[in] k Inner dimension
/// @return Number of add operations (k*n*(n+1)/2)
/// @ingroup flops
inline double fadds_herk( double n, double k )
    { return 0.5*k*n*(n+1); }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for Hermitian rank-2k update.
/// @param[in] n Dimension of Hermitian matrix C
/// @param[in] k Inner dimension
/// @return Number of multiply operations (k*n*n)
/// @ingroup flops
inline double fmuls_her2k( double n, double k )
    { return k*n*n; }

/// @brief Addition count for Hermitian rank-2k update.
/// @param[in] n Dimension of Hermitian matrix C
/// @param[in] k Inner dimension
/// @return Number of add operations (k*n*n)
/// @ingroup flops
inline double fadds_her2k( double n, double k )
    { return k*n*n; }

// -----------------------------------------------------------------------------
/// @brief Multiplication count for triangular matrix-matrix multiply.
/// @param[in] side Side where triangular matrix appears
/// @param[in] m Number of rows of B
/// @param[in] n Number of columns of B
/// @return Number of multiply operations
/// @ingroup flops
inline double fmuls_trmm( blas::Side side, double m, double n )
{
    if (side == blas::Side::Left)
        return 0.5*n*m*(m + 1);
    else
        return 0.5*m*n*(n + 1);
}

/// @brief Addition count for triangular matrix-matrix multiply.
/// @param[in] side Side where triangular matrix appears
/// @param[in] m Number of rows of B
/// @param[in] n Number of columns of B
/// @return Number of add operations
/// @ingroup flops
inline double fadds_trmm( blas::Side side, double m, double n )
{
    if (side == blas::Side::Left)
        return 0.5*n*m*(m - 1);
    else
        return 0.5*m*n*(n - 1);
}

//==============================================================================
/// @brief Data transfer counting in gigabytes.
///
/// Template class for computing data transfer (in gigabytes) for BLAS operations.
/// Accounts for reading and writing matrices/vectors based on operation semantics.
///
/// Example usage:
/// @code
///     double gb = Gbyte<float>::gemm(m, n, k);
///     double gb_complex = Gbyte<std::complex<float>>::gemm(m, n, k);
/// @endcode
///
/// @tparam T Scalar type (e.g., float, double, std::complex<float>)
/// @ingroup flops
template <typename T>
class Gbyte
{
public:
    // ----------------------------------------
    // Level 1 BLAS
    
    /// @brief Data transfer for asum (sum of absolute values).
    /// Reads vector x.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double asum( double n )
        { return 1e-9 * (n * sizeof(T)); }

    /// @brief Data transfer for axpy (y = alpha*x + y).
    /// Reads x and y, writes y.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double axpy( double n )
        { return 1e-9 * (3*n * sizeof(T)); }

    /// @brief Data transfer for copy (y = x).
    /// Reads x, writes y.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double copy( double n )
        { return 1e-9 * (2*n * sizeof(T)); }

    /// @brief Data transfer for iamax (index of max absolute value).
    /// Reads vector x.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double iamax( double n )
        { return 1e-9 * (n * sizeof(T)); }

    /// @brief Data transfer for nrm2 (Euclidean norm).
    /// Reads vector x.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double nrm2( double n )
        { return 1e-9 * (n * sizeof(T)); }

    /// @brief Data transfer for dot product.
    /// Reads vectors x and y.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double dot( double n )
        { return 1e-9 * (2*n * sizeof(T)); }

    /// @brief Data transfer for scal (x = alpha*x).
    /// Reads and writes vector x.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double scal( double n )
        { return 1e-9 * (2*n * sizeof(T)); }

    /// @brief Data transfer for swap (exchange x and y).
    /// Reads and writes vectors x and y.
    /// @param[in] n Vector length
    /// @return Gigabytes transferred
    static double swap( double n )
        { return 1e-9 * (4*n * sizeof(T)); }

    // ----------------------------------------
    // Level 2 BLAS
    
    /// @brief Data transfer for gemv (y = alpha*A*x + beta*y).
    /// Reads matrix A, vectors x and y, writes y.
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @return Gigabytes transferred
    static double gemv( double m, double n )
        { return 1e-9 * ((m*n + m + n) * sizeof(T)); }

    /// @brief Data transfer for hemv (Hermitian matrix-vector multiply).
    /// Reads Hermitian matrix A (triangle), vector x, writes y.
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    static double hemv( double n )
        { return 1e-9 * ((0.5*(n+1)*n + 2*n) * sizeof(T)); }

    /// @brief Data transfer for symv (same as hemv for symmetric).
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    static double symv( double n )
        { return hemv( n ); }

    /// @brief Data transfer for trmv/trsv (triangular matrix-vector ops).
    /// Reads triangular matrix A, vector x, writes x.
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    static double trmv( double n )
        { return 1e-9 * ((0.5*(n+1)*n + 2*n) * sizeof(T)); }

    /// @brief Data transfer for trsv (same as trmv).
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    /// @brief Giga-FLOPs for trsv (triangular solve, same as trmv).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double trsv( double n )
        { return trmv( n ); }

    /// @brief Data transfer for ger (rank-1 update A = A + alpha*x*y^T).
    /// Reads A, x, y, writes A.
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @return Gigabytes transferred
    static double ger( double m, double n )
        { return 1e-9 * ((2*m*n + m + n) * sizeof(T)); }

    /// @brief Data transfer for her/syr (Hermitian/symmetric rank-1 update).
    /// Reads triangular A, vector x, writes triangular A.
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    static double her( double n )
        { return 1e-9 * (((n+1)*n + n) * sizeof(T)); }

    /// @brief Data transfer for syr (same as her for symmetric).
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    /// @brief Giga-FLOPs for syr (symmetric rank-1 update, same as her).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double syr( double n )
        { return her( n ); }

    /// @brief Data transfer for her2/syr2 (Hermitian/symmetric rank-2 update).
    /// Reads triangular A, vectors x and y, writes triangular A.
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    static double her2( double n )
        { return 1e-9 * (((n+1)*n + n + n) * sizeof(T)); }

    /// @brief Data transfer for syr2 (same as her2 for symmetric).
    /// @param[in] n Matrix dimension
    /// @return Gigabytes transferred
    /// @brief Giga-FLOPs for syr2 (symmetric rank-2 update, same as her2).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double syr2( double n )
        { return her2( n ); }

    /// @brief Data transfer for 2D matrix copy.
    /// Reads matrix A, writes matrix B.
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @return Gigabytes transferred
    static double copy_2d( double m, double n )
        { return 1e-9 * (2*m*n * sizeof(T)); }

    // ----------------------------------------
    // Level 3 BLAS
    
    /// @brief Data transfer for gemm (C = alpha*A*B + beta*C).
    /// Reads A, B, C, writes C.
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @param[in] k Inner dimension
    /// @return Gigabytes transferred
    static double gemm( double m, double n, double k )
        { return 1e-9 * ((m*k + k*n + 2*m*n) * sizeof(T)); }

    /// @brief Data transfer for hemm (Hermitian matrix-matrix multiply).
    /// Reads Hermitian A, matrices B and C, writes C.
    /// @param[in] side Side where Hermitian matrix appears
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @return Gigabytes transferred
    static double hemm( blas::Side side, double m, double n )
    {
        // Read Hermitian A, B, C; write C
        double sizeA = (side == blas::Side::Left ? 0.5*m*(m+1) : 0.5*n*(n+1));
        return 1e-9 * ((sizeA + 3*m*n) * sizeof(T));
    }

    /// @brief Data transfer for symm (same as hemm for symmetric).
    /// @param[in] side Side where symmetric matrix appears
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @return Gigabytes transferred
    static double symm( blas::Side side, double m, double n )
        { return hemm( side, m, n ); }

    /// @brief Data transfer for herk (Hermitian rank-k update).
    /// Reads matrix A, Hermitian C, writes C.
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigabytes transferred
    static double herk( double n, double k )
    {
        // Read A, C; write C
        double sizeC = 0.5*n*(n+1);
        return 1e-9 * ((n*k + 2*sizeC) * sizeof(T));
    }

    /// @brief Data transfer for syrk (same as herk for symmetric).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigabytes transferred
    static double syrk( double n, double k )
        { return herk( n, k ); }

    /// @brief Data transfer for her2k (Hermitian rank-2k update).
    /// Reads matrices A and B, Hermitian C, writes C.
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigabytes transferred
    static double her2k( double n, double k )
    {
        // Read A, B, C; write C
        double sizeC = 0.5*n*(n+1);
        return 1e-9 * ((2*n*k + 2*sizeC) * sizeof(T));
    }

    /// @brief Data transfer for syr2k (same as her2k for symmetric).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigabytes transferred
    static double syr2k( double n, double k )
        { return her2k( n, k ); }

    /// @brief Data transfer for trmm/trsm (triangular matrix-matrix ops).
    /// Reads triangular A, matrix B, writes B.
    /// @param[in] side Side where triangular matrix appears
    /// @param[in] m Number of rows of B
    /// @param[in] n Number of columns of B
    /// @return Gigabytes transferred
    static double trmm( blas::Side side, double m, double n )
    {
        // Read triangular A, B; write B
        if (side == blas::Side::Left)
            return 1e-9 * ((0.5*(m+1)*m + 2*m*n) * sizeof(T));
        else
            return 1e-9 * ((0.5*(n+1)*n + 2*m*n) * sizeof(T));
    }

    /// @brief Data transfer for trsm (same as trmm).
    /// @param[in] side Side where triangular matrix appears
    /// @param[in] m Number of rows of B
    /// @param[in] n Number of columns of B
    /// @return Gigabytes transferred
    static double trsm( blas::Side side, double m, double n )
        { return trmm( side, m, n ); }
};

//==============================================================================
/// @brief Traits for counting operations per multiply and add.
///
/// For real types, one multiply = 1 op, one add = 1 op.
/// For complex types, one complex multiply = 6 real ops (4 muls + 2 adds),
/// one complex add = 2 real ops.
///
/// @tparam T Scalar type
/// @ingroup flops
template <typename T>
class FlopTraits
{
public:
    /// @brief Number of real operations for one multiply.
    static constexpr double mul_ops = 1;
    /// @brief Number of real operations for one add.
    static constexpr double add_ops = 1;
};

//------------------------------------------------------------------------------
/// @brief Specialization for complex types.
///
/// Complex multiply: (a+bi)(c+di) = (ac-bd) + (ad+bc)i requires:
/// - 4 real multiplies: ac, bd, ad, bc
/// - 2 real adds: ac-bd, ad+bc
/// Total: 6 real operations
///
/// Complex add: (a+bi) + (c+di) = (a+c) + (b+d)i requires 2 real adds.
///
/// @tparam T Base real type
/// @ingroup flops
template <typename T>
class FlopTraits< std::complex<T> >
{
public:
    /// @brief Number of real operations for one complex multiply (6).
    static constexpr double mul_ops = 6;
    /// @brief Number of real operations for one complex add (2).
    static constexpr double add_ops = 2;
};

//==============================================================================
/// @brief Floating-point operation counting in gigaflops.
///
/// Template class for computing FLOPs (floating-point operations) for BLAS
/// routines. Accounts for both multiplies and adds, properly handling complex
/// arithmetic via FlopTraits.
///
/// Example usage:
/// @code
///     // For single precision real gemm
///     double gflops = Gflop<float>::gemm(m, n, k);
///     
///     // For single precision complex gemm
///     double gflops = Gflop<std::complex<float>>::gemm(m, n, k);
/// @endcode
///
/// @tparam T Scalar type (float, double, std::complex<float>, std::complex<double>)
/// @ingroup flops
template <typename T>
class Gflop
{
public:
    /// @brief Number of real ops per multiply for type T.
    static constexpr double mul_ops = FlopTraits<T>::mul_ops;
    /// @brief Number of real ops per add for type T.
    static constexpr double add_ops = FlopTraits<T>::add_ops;

    // ----------------------------------------
    // Level 1 BLAS
    
    /// @brief Giga-FLOPs for asum (sum of absolute values).
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double asum( double n )
        { return 1e-9 * (mul_ops*fmuls_asum(n) +
                         add_ops*fadds_asum(n)); }

    /// @brief Giga-FLOPs for axpy (y = alpha*x + y).
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double axpy( double n )
        { return 1e-9 * (mul_ops*fmuls_axpy(n) +
                         add_ops*fadds_axpy(n)); }

    /// @brief Giga-FLOPs for copy (no arithmetic operations).
    /// @param[in] n Vector length
    /// @return 0 (copy has no FLOPs)
    static double copy( double n )
        { return 0; }

    /// @brief Giga-FLOPs for iamax (index of maximum absolute value).
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double iamax( double n )
        { return 1e-9 * (mul_ops*fmuls_iamax(n) +
                         add_ops*fadds_iamax(n)); }

    /// @brief Giga-FLOPs for nrm2 (Euclidean norm).
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double nrm2( double n )
        { return 1e-9 * (mul_ops*fmuls_nrm2(n) +
                         add_ops*fadds_nrm2(n)); }

    /// @brief Giga-FLOPs for dot product.
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double dot( double n )
        { return 1e-9 * (mul_ops*fmuls_dot(n) +
                         add_ops*fadds_dot(n)); }

    /// @brief Giga-FLOPs for scal (vector scaling).
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double scal( double n )
        { return 1e-9 * (mul_ops*fmuls_scal(n) +
                         add_ops*fadds_scal(n)); }

    /// @brief Giga-FLOPs for swap (no arithmetic operations).
    /// @param[in] n Vector length
    /// @return 0 (swap has no FLOPs)
    static double swap( double n )
        { return 0; }

    /// @brief Giga-FLOPs for Givens rotation.
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double rot( double n )
        { return 1e-9 * (mul_ops*fmuls_rot(n) +
                         add_ops*fadds_rot(n)); }

    /// @brief Giga-FLOPs for modified Givens rotation.
    /// @param[in] n Vector length
    /// @return Gigaflops
    static double rotm( double n )
        { return 1e-9 * (mul_ops*fmuls_rotm(n) +
                         add_ops*fadds_rotm(n)); }

    // ----------------------------------------
    // Level 2 BLAS
    
    /// @brief Giga-FLOPs for gemv (general matrix-vector multiply).
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @return Gigaflops
    static double gemv(double m, double n)
        { return 1e-9 * (mul_ops*fmuls_gemv(m, n) +
                         add_ops*fadds_gemv(m, n)); }

    /// @brief Giga-FLOPs for symv (symmetric matrix-vector multiply).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double symv(double n)
        { return gemv( n, n ); }

    /// @brief Giga-FLOPs for hemv (Hermitian matrix-vector multiply).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double hemv(double n)
        { return symv( n ); }

    /// @brief Giga-FLOPs for trmv (triangular matrix-vector multiply).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double trmv( double n )
        { return 1e-9 * (mul_ops*fmuls_trmv(n) +
                         add_ops*fadds_trmv(n)); }

    static double trsv( double n )
        { return trmv( n ); }

    /// @brief Giga-FLOPs for her (Hermitian rank-1 update).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double her( double n )
        { return ger( n, n ); }

    static double syr( double n )
        { return her( n ); }

    /// @brief Giga-FLOPs for ger (general rank-1 update).
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @return Gigaflops
    static double ger( double m, double n )
        { return 1e-9 * (mul_ops*fmuls_ger(m, n) +
                         add_ops*fadds_ger(m, n)); }

    /// @brief Giga-FLOPs for her2 (Hermitian rank-2 update).
    /// @param[in] n Matrix dimension
    /// @return Gigaflops
    static double her2( double n )
        { return 2*ger( n, n ); }

    static double syr2( double n )
        { return her2( n ); }

    // ----------------------------------------
    // Level 3 BLAS
    /// @brief Giga-FLOPs for gemm (C = alpha*op(A)*op(B) + beta*C).
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @param[in] k Inner dimension
    /// @return Gigaflops
    static double gemm(double m, double n, double k)
        { return 1e-9 * (mul_ops*fmuls_gemm(m, n, k) +
                         add_ops*fadds_gemm(m, n, k)); }

    /// @brief Giga-FLOPs for gbmm (banded matrix-matrix multiply).
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @param[in] k Inner dimension
    /// @param[in] kl Lower bandwidth
    /// @param[in] ku Upper bandwidth
    /// @return Gigaflops
    static double gbmm(double m, double n, double k, double kl, double ku)
        { return 1e-9 * (mul_ops*fmuls_gbmm(m, n, k, kl, ku) +
                         add_ops*fadds_gbmm(m, n, k, kl, ku)); }

    /// @brief Giga-FLOPs for hemm (Hermitian matrix-matrix multiply).
    /// @param[in] side Side where Hermitian matrix appears
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @return Gigaflops
    static double hemm(blas::Side side, double m, double n)
        { return 1e-9 * (mul_ops*fmuls_hemm(side, m, n) +
                         add_ops*fadds_hemm(side, m, n)); }

    /// @brief Giga-FLOPs for hbmm (Hermitian banded matrix-matrix multiply).
    /// @param[in] m Number of rows
    /// @param[in] n Number of columns
    /// @param[in] kd Bandwidth
    /// @return Gigaflops
    static double hbmm(double m, double n, double kd)
        { return gbmm(m, n, m, kd, kd); }

    /// @brief Giga-FLOPs for symm (symmetric matrix-matrix multiply).
    /// @param[in] side Side where symmetric matrix appears
    /// @param[in] m Number of rows of C
    /// @param[in] n Number of columns of C
    /// @return Gigaflops
    static double symm(blas::Side side, double m, double n)
        { return hemm( side, m, n ); }

    /// @brief Giga-FLOPs for herk (Hermitian rank-k update).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigaflops
    static double herk(double n, double k)
        { return 1e-9 * (mul_ops*fmuls_herk(n, k) +
                         add_ops*fadds_herk(n, k)); }

    /// @brief Giga-FLOPs for syrk (symmetric rank-k update, same as herk).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigaflops
    static double syrk(double n, double k)
        { return herk( n, k ); }

    /// @brief Giga-FLOPs for her2k (Hermitian rank-2k update).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigaflops
    static double her2k(double n, double k)
        { return 1e-9 * (mul_ops*fmuls_her2k(n, k) +
                         add_ops*fadds_her2k(n, k)); }

    /// @brief Giga-FLOPs for syr2k (symmetric rank-2k update, same as her2k).
    /// @param[in] n Dimension of C
    /// @param[in] k Inner dimension
    /// @return Gigaflops
    static double syr2k(double n, double k)
        { return her2k( n, k ); }

    /// @brief Giga-FLOPs for trmm (triangular matrix-matrix multiply).
    /// @param[in] side Side where triangular matrix appears
    /// @param[in] m Number of rows of B
    /// @param[in] n Number of columns of B
    /// @return Gigaflops
    static double trmm(blas::Side side, double m, double n)
        { return 1e-9 * (mul_ops*fmuls_trmm(side, m, n) +
                         add_ops*fadds_trmm(side, m, n)); }

    /// @brief Giga-FLOPs for trsm (triangular solve, same as trmm).
    /// @param[in] side Side where triangular matrix appears
    /// @param[in] m Number of rows of B
    /// @param[in] n Number of columns of B
    /// @return Gigaflops
    static double trsm(blas::Side side, double m, double n)
        { return trmm( side, m, n ); }

};

}  // namespace blas

#endif        //  #ifndef BLAS_FLOPS_HH
