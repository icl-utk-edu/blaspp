// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_COUNTER_HH
#define BLAS_COUNTER_HH

#include "blas/defines.h"
#include "blas/util.hh"
#include "blas/flops.hh"
#include <atomic>

#ifdef BLAS_HAVE_PAPI
    #include "sde_lib.h"
    #include "sde_lib.hpp"
#endif

namespace blas {

//==============================================================================
/// @brief Performance counter integration for BLAS++.
///
/// This class provides integration with PAPI (Performance API) for counting
/// BLAS operations and computing floating-point operation counts. Uses the
/// Scott Meyers Singleton pattern for thread-safe initialization.
///
/// The counter system tracks:
/// - Number of calls to each BLAS routine
/// - Dimensions and parameters for each call
/// - Total floating-point operations performed
///
/// Usage (when PAPI is available):
/// @code
///     // Insert operation into counting set
///     counter::gemm_type op = {transA, transB, m, n, k};
///     counter::insert(op, counter::Id::gemm);
///     
///     // Get total flop count
///     long long flops = counter::get_flop_count(&atomic_var);
/// @endcode
///
/// @note This is essentially a namespace - all public functions are static.
/// @ingroup util
class counter
{
public:
    #ifdef BLAS_HAVE_PAPI
        typedef papi_sde::PapiSde::CountingSet CountingSet;
    #else
        typedef void CountingSet;
        typedef void cset_list_object_t;
    #endif

public:
    //------------------------------------------------------------------------------
    /// @brief Operation identifiers for the counter set.
    ///
    /// These IDs differentiate BLAS routines in the counter. Separate IDs
    /// exist for CPU (host) and device (GPU) versions of each routine.
    ///
    /// @ingroup util
    enum class Id {
        // Level 1 BLAS
        asum,    ///< Sum of absolute values
        axpy,    ///< Y = alpha*X + Y
        copy,    ///< Copy vector
        dot,     ///< Dot product (conjugated)
        dotu,    ///< Dot product (unconjugated)
        iamax,   ///< Index of maximum absolute value
        nrm2,    ///< Euclidean norm
        rot,     ///< Givens rotation
        rotg,    ///< Generate Givens rotation
        rotm,    ///< Modified Givens rotation
        rotmg,   ///< Generate modified Givens rotation
        scal,    ///< Scale vector
        swap,    ///< Swap vectors

        // Level 2 BLAS
        gemv,    ///< General matrix-vector multiply
        ger,     ///< General rank-1 update (conjugated)
        geru,    ///< General rank-1 update (unconjugated)
        hemv,    ///< Hermitian matrix-vector multiply
        her,     ///< Hermitian rank-1 update
        her2,    ///< Hermitian rank-2 update
        symv,    ///< Symmetric matrix-vector multiply
        syr,     ///< Symmetric rank-1 update
        syr2,    ///< Symmetric rank-2 update
        trmv,    ///< Triangular matrix-vector multiply
        trsv,    ///< Triangular solve

        // Level 3 BLAS
        gemm,    ///< General matrix-matrix multiply
        hemm,    ///< Hermitian matrix-matrix multiply
        herk,    ///< Hermitian rank-k update
        her2k,   ///< Hermitian rank-2k update
        symm,    ///< Symmetric matrix-matrix multiply
        syrk,    ///< Symmetric rank-k update
        syr2k,   ///< Symmetric rank-2k update
        trmm,    ///< Triangular matrix-matrix multiply
        trsm,    ///< Triangular solve with multiple RHS

        // Device (GPU) Level 1 BLAS
        dev_asum,   ///< Device asum
        dev_axpy,   ///< Device axpy
        dev_copy,   ///< Device copy
        dev_dot,    ///< Device dot
        dev_dotu,   ///< Device dotu
        dev_iamax,  ///< Device iamax
        dev_nrm2,   ///< Device nrm2
        dev_rot,    ///< Device rotation
        dev_rotg,   ///< Device rotation generation
        dev_rotm,   ///< Device modified rotation
        dev_rotmg,  ///< Device modified rotation generation
        dev_scal,   ///< Device scale
        dev_swap,   ///< Device swap

        // Device (GPU) Level 2 BLAS
        dev_gemv,   ///< Device gemv
        dev_ger,    ///< Device ger
        dev_geru,   ///< Device geru
        dev_hemv,   ///< Device hemv
        dev_her,    ///< Device her
        dev_her2,   ///< Device her2
        dev_symv,   ///< Device symv
        dev_syr,    ///< Device syr
        dev_syr2,   ///< Device syr2
        dev_trmv,   ///< Device trmv
        dev_trsv,   ///< Device trsv

        // Device (GPU) Level 3 BLAS
        dev_gemm,    ///< Device gemm
        dev_hemm,    ///< Device hemm
        dev_herk,    ///< Device herk
        dev_her2k,   ///< Device her2k
        dev_symm,    ///< Device symm
        dev_syrk,    ///< Device syrk
        dev_syr2k,   ///< Device syr2k
        dev_trmm,    ///< Device trmm
        dev_trsm,    ///< Device trsm

        // Device batch BLAS
        dev_batch_gemm,  ///< Device batch gemm
        dev_batch_hemm,  ///< Device batch hemm

    };

    //==============================================================================
    // Operation parameter structures
    // These structs store the dimensions and parameters for each BLAS call.
    
    /// @brief Parameters for Level 1 BLAS operations (vector length only).
    /// Used by: axpy, scal, copy, swap, dot, dotu, nrm2, asum, iamax, rot, rotm.
    /// @ingroup util
    struct axpy_type {
        int64_t n;  ///< Vector length
    };

    typedef axpy_type scal_type;
    typedef axpy_type copy_type;
    typedef axpy_type swap_type;
    typedef axpy_type dot_type;
    typedef axpy_type dotu_type;
    typedef axpy_type nrm2_type;
    typedef axpy_type asum_type;
    typedef axpy_type iamax_type;
    typedef axpy_type rot_type;
    typedef axpy_type rotm_type;
    typedef axpy_type rotg_type;
    typedef axpy_type rotmg_type;

    //==============================================================================
    // Level 2 BLAS

    /// @brief Parameters for gemv (general matrix-vector multiply).
    /// @ingroup util
    struct gemv_type {
        blas::Op trans;  ///< Transpose operation
        int64_t m, n;    ///< Matrix dimensions
    };

    //------------------------------------------------------------------------------
    /// @brief Parameters for Hermitian/symmetric matrix-vector operations.
    /// Used by: hemv, symv, her, her2, syr, syr2.
    /// @ingroup util
    struct hemv_type {
        blas::Uplo uplo;  ///< Upper or lower triangle
        int64_t n;        ///< Matrix dimension
    };

    typedef hemv_type symv_type;
    typedef hemv_type her_type;
    typedef hemv_type her2_type;
    typedef hemv_type syr_type;
    typedef hemv_type syr2_type;

    //------------------------------------------------------------------------------
    /// @brief Parameters for triangular matrix-vector operations.
    /// Used by: trmv, trsv.
    /// @ingroup util
    struct trmv_type {
        blas::Uplo uplo;  ///< Upper or lower triangle
        blas::Op trans;   ///< Transpose operation
        blas::Diag diag;  ///< Unit or non-unit diagonal
        int64_t n;        ///< Matrix dimension
    };

    typedef trmv_type trsv_type;

    //------------------------------------------------------------------------------
    /// @brief Parameters for rank-1 update operations.
    /// Used by: ger, geru, gerc.
    /// @ingroup util
    struct ger_type {
        int64_t m, n;  ///< Matrix dimensions
    };

    typedef ger_type geru_type;
    typedef ger_type gerc_type;

    //==============================================================================
    // Level 3 BLAS parameter structures

    /// @brief Parameters for gemm (general matrix-matrix multiply).
    /// @ingroup util
    struct gemm_type {
        blas::Op transA, transB;  ///< Transpose operations for A and B
        int64_t m, n, k;          ///< Matrix dimensions
    };

    //------------------------------------------------------------------------------
    /// @brief Parameters for Hermitian/symmetric matrix-matrix multiply.
    /// Used by: hemm, symm.
    /// @ingroup util
    struct hemm_type {
        blas::Side side;  ///< Side where Hermitian/symmetric matrix appears
        blas::Uplo uplo;  ///< Upper or lower triangle
        int64_t m, n;     ///< Matrix dimensions
    };

    typedef hemm_type symm_type;

    //------------------------------------------------------------------------------
    /// @brief Parameters for Hermitian/symmetric rank-k and rank-2k updates.
    /// Used by: herk, syrk, syr2k, her2k.
    /// @ingroup util
    struct herk_type {
        blas::Uplo uplo;  ///< Upper or lower triangle of result
        blas::Op trans;   ///< Transpose operation
        int64_t n, k;     ///< Matrix dimensions
    };

    typedef herk_type syrk_type;
    typedef herk_type syr2k_type;
    typedef herk_type her2k_type;

    //------------------------------------------------------------------------------
    /// @brief Parameters for triangular matrix-matrix operations.
    /// Used by: trmm, trsm.
    /// @ingroup util
    struct trmm_type {
        blas::Side side;  ///< Side where triangular matrix appears
        blas::Uplo uplo;  ///< Upper or lower triangle
        blas::Op transA;  ///< Transpose operation
        blas::Diag diag;  ///< Unit or non-unit diagonal
        int64_t m, n;     ///< Matrix dimensions
    };

    typedef trmm_type trsm_type;

    //==============================================================================
    // Device BLAS parameter structures
    // Type aliases for device operations (same parameters as host versions)
    
    typedef axpy_type dev_axpy_type;  ///< Device axpy parameters
    typedef axpy_type dev_scal_type;
    typedef axpy_type dev_copy_type;
    typedef axpy_type dev_swap_type;
    typedef axpy_type dev_dot_type;
    typedef axpy_type dev_dotu_type;
    typedef axpy_type dev_nrm2_type;
    typedef axpy_type dev_asum_type;
    typedef axpy_type dev_iamax_type;
    typedef axpy_type dev_rot_type;
    typedef axpy_type dev_rotm_type;
    typedef axpy_type dev_rotg_type;
    typedef axpy_type dev_rotmg_type;

    //------------------------------------------------------------------------------
    typedef gemv_type dev_gemv_type;

    typedef hemv_type dev_hemv_type;
    typedef hemv_type dev_symv_type;
    typedef hemv_type dev_her_type;
    typedef hemv_type dev_her2_type;
    typedef hemv_type dev_syr_type;
    typedef hemv_type dev_syr2_type;

    typedef trmv_type dev_trmv_type;
    typedef trmv_type dev_trsv_type;

    typedef ger_type dev_ger_type;
    typedef ger_type dev_geru_type;
    typedef ger_type dev_gerc_type;

    //------------------------------------------------------------------------------
    typedef gemm_type dev_gemm_type;

    typedef hemm_type dev_hemm_type;
    typedef hemm_type dev_symm_type;

    typedef herk_type dev_herk_type;
    typedef herk_type dev_syrk_type;
    typedef herk_type dev_syr2k_type;
    typedef herk_type dev_her2k_type;

    typedef trmm_type dev_trmm_type;
    typedef trmm_type dev_trsm_type;

    //==============================================================================
    // Device batch BLAS parameter structures

    /// @brief Parameters for batch gemm on device.
    /// @ingroup util
    struct dev_batch_gemm_type {
        blas::Op transA, transB;  ///< Transpose operations
        int64_t m, n, k;          ///< Matrix dimensions
        size_t batch_size;        ///< Number of matrices in batch
    };

    //------------------------------------------------------------------------------
    /// @brief Parameters for batch hemm on device.
    /// @ingroup util
    struct dev_batch_hemm_type {
        size_t batch_size;  ///< Number of matrices in batch
    };

    //==============================================================================
    // Public API for counter operations
    
    //--------------------------------------------------------------------------
    /// @brief Get singleton instance of counter.
    ///
    /// Initializes PAPI counters on first call using Scott Meyers singleton pattern.
    /// Thread-safe initialization guaranteed by C++11.
    ///
    /// @return Reference to counter singleton
    /// @ingroup util
    static counter &get()
    {
        static counter s_cnt;
        return s_cnt;
    }

    //--------------------------------------------------------------------------
    /// @brief Insert operation into the PAPI counting set.
    ///
    /// Records an operation with its parameters for later analysis. When PAPI
    /// is not available, this is a no-op.
    ///
    /// Example:
    /// @code
    ///     counter::gemm_type op = {transA, transB, m, n, k};
    ///     counter::insert(op, counter::Id::gemm);
    /// @endcode
    ///
    /// @param[in] element Operation parameters (e.g., gemm_type, axpy_type)
    /// @param[in] id Operation identifier from Id enum
    /// @tparam T Parameter structure type
    /// @ingroup util
    template <typename T>
    static void insert( T element, Id id )
    {
        #ifdef BLAS_HAVE_PAPI
            get().set_->insert( element, uint32_t( id ) );
        #endif
    }

    //--------------------------------------------------------------------------
    /// @brief Insert operation with custom hashable size.
    ///
    /// Advanced version allowing control over which portion of the parameter
    /// structure is used for hashing. Useful when some fields should not
    /// affect operation grouping.
    ///
    /// @param[in] hashable_size Number of bytes to use for hashing (≤ sizeof(element))
    /// @param[in] element Operation parameters
    /// @param[in] id Operation identifier from Id enum
    /// @tparam T Parameter structure type
    /// @ingroup util
    template <typename T>
    static void insert( size_t hashable_size, T element, Id id )
    {
        #ifdef BLAS_HAVE_PAPI
            get().set_->insert( hashable_size, element, uint32_t( id ) );
        #endif
    }

    //--------------------------------------------------------------------------
    /// @brief Get current total FLOP count.
    ///
    /// Returns cumulative floating-point operations across all recorded BLAS calls.
    /// When PAPI is not available, returns 0.
    ///
    /// @param[in] atmc_var Pointer to atomic variable holding flop count
    /// @return Total FLOP count
    /// @ingroup util
    static long long int get_flop_count(std::atomic<long long> *atmc_var)
    {
        long long int fp = 0;
        #ifdef BLAS_HAVE_PAPI
            fp = *atmc_var;
        #endif
        return fp;
    }

    //--------------------------------------------------------------------------
    /// @brief Increment total FLOP count.
    ///
    /// Adds to the cumulative FLOP counter. Thread-safe via atomic operations.
    /// When PAPI is not available, this is a no-op.
    ///
    /// @param[in] fp Number of FLOPs to add
    /// @ingroup util
    static void inc_flop_count(long long int fp)
    {
        #ifdef BLAS_HAVE_PAPI
            get().total_flop_count_ += fp;
        #endif
        return;
    }

    //--------------------------------------------------------------------------
    /// @brief Print detailed operation statistics.
    ///
    /// Outputs comprehensive listing of all recorded operations with:
    /// - Operation parameters
    /// - Number of calls
    /// - FLOP count per operation type
    /// - Total cumulative FLOPs
    ///
    /// When PAPI is not available, this is a no-op.
    ///
    /// @param[in] list Linked list of counting set objects from PAPI
    /// @ingroup util
    static void print( cset_list_object_t* list )
    {
        #ifdef BLAS_HAVE_PAPI
            double totalflops = 0;
            for (auto iter = list; iter != nullptr; iter = iter->next) {
                Id type_id = static_cast<Id>( iter->type_id );
                switch (type_id) {
                    // Level 1 BLAS
                    case Id::axpy: {
                        auto *ptr = static_cast<axpy_type *>( iter->ptr );
                        double flop = Gflop<double>::axpy( ptr->n ) * 1e9 * iter->count;
                        printf( "axpy( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::scal: {
                        auto *ptr = static_cast<scal_type *>( iter->ptr );
                        double flop = Gflop<double>::scal( ptr->n ) * 1e9 * iter->count;
                        printf( "scal( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::copy: {
                        auto *ptr = static_cast<copy_type *>( iter->ptr );
                        double flop = Gflop<double>::copy( ptr->n ) * 1e9 * iter->count;
                        printf( "copy( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::swap: {
                        auto *ptr = static_cast<swap_type *>( iter->ptr );
                        double flop = Gflop<double>::swap( ptr->n ) * 1e9 * iter->count;
                        printf( "swap( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dot: {
                        auto *ptr = static_cast<dot_type *>( iter->ptr );
                        double flop = Gflop<double>::dot( ptr->n ) * 1e9 * iter->count;
                        printf( "dot( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dotu: {
                        auto *ptr = static_cast<dotu_type *>( iter->ptr );
                        double flop = Gflop<double>::dot( ptr->n ) * 1e9 * iter->count;
                        printf( "dotu( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::nrm2: {
                        auto *ptr = static_cast<nrm2_type *>( iter->ptr );
                        double flop = Gflop<double>::nrm2( ptr->n ) * 1e9 * iter->count;
                        printf( "nrm2( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::asum: {
                        auto *ptr = static_cast<asum_type *>( iter->ptr );
                        double flop = Gflop<double>::asum( ptr->n ) * 1e9 * iter->count;
                        printf( "asum( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::iamax: {
                        auto *ptr = static_cast<iamax_type *>( iter->ptr );
                        double flop = Gflop<double>::iamax( ptr->n ) * 1e9 * iter->count;
                        printf( "iamax( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::rotg: {
                        // auto *ptr = static_cast<rotg_type *>( iter->ptr );
                        // double flop = Gflop<double>::rotg( ptr->n ) * 1e9;
                        printf( "rotg( ) count %d\n", iter->count );
                        // totalflops += flop;
                        break;
                    }
                    case Id::rot: {
                        auto *ptr = static_cast<rot_type *>( iter->ptr );
                        double flop = Gflop<double>::rot( ptr->n ) * 1e9 * iter->count;
                        printf( "rot( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::rotmg: {
                        // auto *ptr = static_cast<rotmg_type *>( iter->ptr );
                        // double flop = Gflop<double>::rotmg( ptr->n ) * 1e9;
                        printf( "rotmg( ) count %d\n", iter->count );
                        // totalflops += flop;
                        break;
                    }
                    case Id::rotm: {
                        auto *ptr = static_cast<rotm_type *>( iter->ptr );
                        double flop = Gflop<double>::rotm( ptr->n ) * 1e9 * iter->count;
                        printf( "rotm( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 2 BLAS
                    case Id::gemv: {
                        auto *ptr = static_cast<gemv_type *>( iter->ptr );
                        double flop = Gflop<double>::gemv( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "gemv( %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->trans ), llong( ptr->m ), llong( ptr->n ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemv: {
                        auto *ptr = static_cast<hemv_type *>( iter->ptr );
                        double flop = Gflop<double>::hemv( ptr->n ) * 1e9 * iter->count;
                        printf( "hemv( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symv: {
                        auto *ptr = static_cast<symv_type *>( iter->ptr );
                        double flop = Gflop<double>::symv( ptr->n ) * 1e9 * iter->count;
                        printf( "symv( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmv: {
                        auto *ptr = static_cast<trmv_type *>( iter->ptr );
                        double flop = Gflop<double>::trmv( ptr->n ) * 1e9 * iter->count;
                        printf( "trmv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                to_char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsv: {
                        auto *ptr = static_cast<trsv_type *>( iter->ptr );
                        double flop = Gflop<double>::trsv( ptr->n ) * 1e9 * iter->count;
                        printf( "trsv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                to_char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::ger: {
                        auto *ptr = static_cast<ger_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "ger( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::geru: {
                        auto *ptr = static_cast<geru_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "geru( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her: {
                        auto *ptr = static_cast<her_type *>( iter->ptr );
                        double flop = Gflop<double>::her( ptr->n ) * 1e9 * iter->count;
                        printf( "her( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2: {
                        auto *ptr = static_cast<her_type *>( iter->ptr );
                        double flop = Gflop<double>::her2( ptr->n ) * 1e9 * iter->count;
                        printf( "her2( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr: {
                        auto *ptr = static_cast<syr_type *>( iter->ptr );
                        double flop = Gflop<double>::syr( ptr->n ) * 1e9 * iter->count;
                        printf( "syr( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2: {
                        auto *ptr = static_cast<syr2_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2( ptr->n ) * 1e9 * iter->count;
                        printf( "syr2( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 3 BLAS
                    case Id::gemm: {
                        auto *ptr = static_cast<gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "gemm( %c, %c, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->transA ), to_char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemm: {
                        auto *ptr = static_cast<hemm_type *>( iter->ptr );
                        double flop = Gflop<double>::hemm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "hemm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2k: {
                        auto *ptr = static_cast<her2k_type *>( iter->ptr );
                        double flop = Gflop<double>::her2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "her2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::herk: {
                        auto *ptr = static_cast<herk_type *>( iter->ptr );
                        double flop = Gflop<double>::herk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "herk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symm: {
                        auto *ptr = static_cast<symm_type *>( iter->ptr );
                        double flop = Gflop<double>::symm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "symm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2k: {
                        auto *ptr = static_cast<syr2k_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "syr2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syrk: {
                        auto *ptr = static_cast<syrk_type *>( iter->ptr );
                        double flop = Gflop<double>::syrk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "syrk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmm: {
                        auto *ptr = static_cast<trmm_type *>( iter->ptr );
                        double flop = Gflop<double>::trmm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "trmm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                to_char( ptr->transA ), to_char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsm: {
                        auto *ptr = static_cast<trsm_type *>( iter->ptr );
                        double flop = Gflop<double>::trsm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "trsm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                to_char( ptr->transA ), to_char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 1 Device BLAS
                    case Id::dev_axpy: {
                        auto *ptr = static_cast<dev_axpy_type *>( iter->ptr );
                        double flop = Gflop<double>::axpy( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_axpy( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_scal: {
                        auto *ptr = static_cast<dev_scal_type *>( iter->ptr );
                        double flop = Gflop<double>::scal( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_scal( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_copy: {
                        auto *ptr = static_cast<dev_copy_type *>( iter->ptr );
                        double flop = Gflop<double>::copy( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_copy( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_swap: {
                        auto *ptr = static_cast<dev_swap_type *>( iter->ptr );
                        double flop = Gflop<double>::swap( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_swap( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_dot: {
                        auto *ptr = static_cast<dev_dot_type *>( iter->ptr );
                        double flop = Gflop<double>::dot( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_dot( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_dotu: {
                        auto *ptr = static_cast<dev_dotu_type *>( iter->ptr );
                        double flop = Gflop<double>::dot( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_dotu( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_nrm2: {
                        auto *ptr = static_cast<dev_nrm2_type *>( iter->ptr );
                        double flop = Gflop<double>::nrm2( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_nrm2( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_asum: {
                        auto *ptr = static_cast<dev_asum_type *>( iter->ptr );
                        double flop = Gflop<double>::asum( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_asum( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_iamax: {
                        auto *ptr = static_cast<dev_iamax_type *>( iter->ptr );
                        double flop = Gflop<double>::iamax( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_iamax( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_rotg: {
                        // auto *ptr = static_cast<dev_rotg_type *>( iter->ptr );
                        // double flop = Gflop<double>::rotg( ptr->n ) * 1e9;
                        printf( "dev_rotg( ) count %d\n", iter->count );
                        // totalflops += flop;
                        break;
                    }
                    case Id::dev_rot: {
                        auto *ptr = static_cast<dev_rot_type *>( iter->ptr );
                        double flop = Gflop<double>::rot( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_rot( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_rotmg: {
                        // auto *ptr = static_cast<dev_rotmg_type *>( iter->ptr );
                        // double flop = Gflop<double>::rotmg( ptr->n ) * 1e9;
                        printf( "dev_rotmg( ) count %d\n", iter->count );
                        // totalflops += flop;
                        break;
                    }
                    case Id::dev_rotm: {
                        auto *ptr = static_cast<dev_rotm_type *>( iter->ptr );
                        double flop = Gflop<double>::rotm( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_rotm( %lld ) count %d, flop count %.2e\n",
                                llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 2 Device BLAS
                    case Id::dev_gemv: {
                        auto *ptr = static_cast<dev_gemv_type *>( iter->ptr );
                        double flop = Gflop<double>::gemv( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_gemv( %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->trans ), llong( ptr->m ), llong( ptr->n ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_hemv: {
                        auto *ptr = static_cast<dev_hemv_type *>( iter->ptr );
                        double flop = Gflop<double>::hemv( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_hemv( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_symv: {
                        auto *ptr = static_cast<dev_symv_type *>( iter->ptr );
                        double flop = Gflop<double>::symv( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_symv( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trmv: {
                        auto *ptr = static_cast<dev_trmv_type *>( iter->ptr );
                        double flop = Gflop<double>::trmv( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trmv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                to_char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trsv: {
                        auto *ptr = static_cast<dev_trsv_type *>( iter->ptr );
                        double flop = Gflop<double>::trsv( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trsv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                to_char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_ger: {
                        auto *ptr = static_cast<dev_ger_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_ger( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_geru: {
                        auto *ptr = static_cast<dev_geru_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_geru( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_her: {
                        auto *ptr = static_cast<dev_her_type *>( iter->ptr );
                        double flop = Gflop<double>::her( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_her( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_her2: {
                        auto *ptr = static_cast<dev_her_type *>( iter->ptr );
                        double flop = Gflop<double>::her2( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_her2( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syr: {
                        auto *ptr = static_cast<dev_syr_type *>( iter->ptr );
                        double flop = Gflop<double>::syr( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_syr( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syr2: {
                        auto *ptr = static_cast<dev_syr2_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_syr2( %c, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 3 Device BLAS
                    case Id::dev_gemm: {
                        auto *ptr = static_cast<dev_gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_gemm( %c, %c, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->transA ), to_char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_hemm: {
                        auto *ptr = static_cast<dev_hemm_type *>( iter->ptr );
                        double flop = Gflop<double>::hemm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_hemm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_her2k: {
                        auto *ptr = static_cast<dev_her2k_type *>( iter->ptr );
                        double flop = Gflop<double>::her2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_her2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_herk: {
                        auto *ptr = static_cast<dev_herk_type *>( iter->ptr );
                        double flop = Gflop<double>::herk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_herk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_symm: {
                        auto *ptr = static_cast<dev_symm_type *>( iter->ptr );
                        double flop = Gflop<double>::symm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_symm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syr2k: {
                        auto *ptr = static_cast<dev_syr2k_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_syr2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syrk: {
                        auto *ptr = static_cast<dev_syrk_type *>( iter->ptr );
                        double flop = Gflop<double>::syrk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_syrk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->uplo ), to_char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trmm: {
                        auto *ptr = static_cast<dev_trmm_type *>( iter->ptr );
                        double flop = Gflop<double>::trmm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trmm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                to_char( ptr->transA ), to_char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trsm: {
                        auto *ptr = static_cast<dev_trsm_type *>( iter->ptr );
                        double flop = Gflop<double>::trsm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trsm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->side ), to_char( ptr->uplo ),
                                to_char( ptr->transA ), to_char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Device batch BLAS
                    case Id::dev_batch_gemm: {
                        auto *ptr = static_cast<dev_batch_gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_batch_gemm( %c, %c, %lld, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                to_char( ptr->transA ), to_char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                llong( ptr->batch_size ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_batch_hemm: {
                        auto *ptr = static_cast<dev_batch_hemm_type *>( iter->ptr );
                        printf( "dev_batch_hemm( ) batch count %lld, count %d\n",
                                llong( ptr->batch_size ), iter->count );
                        break;
                    }
                }
            }
            printf( "total BLAS flop count %.2e\n", totalflops );
        #endif
    }

private:
    //--------------------------------------------------------------------------
    /// @brief Private constructor for singleton pattern.
    ///
    /// Initializes PAPI Software Defined Events (SDEs) on first call.
    /// Creates counting set and registers FLOP counter callback.
    counter()
    {
        set_ = nullptr;
        total_flop_count_ = -1;
        #ifdef BLAS_HAVE_PAPI
            papi_sde::PapiSde sde( "blas" );
            set_ = sde.create_counting_set( "counter" );
            total_flop_count_ = 0;
            sde.register_counter_cb("flops", PAPI_SDE_RO|PAPI_SDE_DELTA, get_flop_count, total_flop_count_);
        #endif
    }

    CountingSet* set_;
    std::atomic<long long> total_flop_count_;
};  // class count

}  // namespace blas

#endif  // BLAS_COUNTER_HH
