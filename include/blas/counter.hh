// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_COUNTER_HH
#define BLAS_COUNTER_HH

#include "blas/defines.h"
#include "blas/util.hh"
#include "blas/flops.hh"

#ifdef BLAS_HAVE_PAPI
    #include "sde_lib.h"
    #include "sde_lib.hpp"
#endif

namespace blas {

//==============================================================================
/// Initialize PAPI counters for BLAS++.
/// Uses thread-safe Scott Meyers Singleton.
/// This class acts like a namespace -- all public functions are static.
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
    /// ID to differentiate routines in the counter set.
    enum class Id {
        // Level 1 BLAS
        asum,
        axpy,
        copy,
        dot,
        dotu,
        iamax,
        nrm2,
        rot,
        rotg,
        rotm,
        rotmg,
        scal,
        swap,

        // Level 2 BLAS
        gemv,
        ger,
        geru,
        hemv,
        her,
        her2,
        symv,
        syr,
        syr2,
        trmv,
        trsv,

        // Level 3 BLAS
        gemm,
        hemm,
        herk,
        her2k,
        symm,
        syrk,
        syr2k,
        trmm,
        trsm,

        // Device BLAS
        dev_copy,
        dev_dot,
        dev_gemm,
        dev_hemm,
        dev_her2k,
        dev_herk,
        dev_nrm2,
        dev_scal,
        dev_swap,
        dev_symm,
        dev_syr2k,
        dev_syrk,
        dev_trmm,
        dev_trsm,

        // Device batch BLAS
        dev_batch_gemm,
        dev_batch_hemm,

    };

    //==============================================================================
    // Level 1 BLAS

    struct axpy_type {
        int64_t n;
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

    struct gemv_type {
        blas::Op trans;
        int64_t m, n;
    };

    //------------------------------------------------------------------------------
    struct hemv_type {
        blas::Uplo uplo;
        int64_t n;
    };

    typedef hemv_type symv_type;
    typedef hemv_type her_type;
    typedef hemv_type her2_type;
    typedef hemv_type syr_type;
    typedef hemv_type syr2_type;

    //------------------------------------------------------------------------------
    struct trmv_type {
        blas::Uplo uplo;
        blas::Op trans;
        blas::Diag diag;
        int64_t n;
    };

    typedef trmv_type trsv_type;

    //------------------------------------------------------------------------------
    struct ger_type {
        int64_t m, n;
    };

    typedef ger_type geru_type;
    typedef ger_type gerc_type;

    //==============================================================================
    // Level 3 BLAS

    struct gemm_type {
        blas::Op transA, transB;
        int64_t m, n, k;
    };

    //------------------------------------------------------------------------------
    struct hemm_type {
        blas::Side side;
        blas::Uplo uplo;
        int64_t m, n;
    };

    typedef hemm_type symm_type;

    //------------------------------------------------------------------------------
    struct herk_type {
        blas::Uplo uplo;
        blas::Op trans;
        int64_t n, k;
    };

    typedef herk_type syrk_type;
    typedef herk_type syr2k_type;
    typedef herk_type her2k_type;

    //------------------------------------------------------------------------------
    struct trmm_type {
        blas::Side side;
        blas::Uplo uplo;
        blas::Op transA;
        blas::Diag diag;
        int64_t m, n;
    };

    typedef trmm_type trsm_type;

    //==============================================================================
    // Device BLAS

    typedef axpy_type dev_copy_type;
    typedef axpy_type dev_dot_type;
    typedef axpy_type dev_nrm2_type;
    typedef axpy_type dev_scal_type;
    typedef axpy_type dev_swap_type;

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
    // Device batch BLAS

    struct dev_batch_gemm_type {
        blas::Op transA, transB;
        int64_t m, n, k;
        size_t batch_size;
    };

    //------------------------------------------------------------------------------
    struct dev_batch_hemm_type {
        size_t batch_size;
    };

    //--------------------------------------------------------------------------
    /// Initializes PAPI counting set on first call.
    /// Without PAPI, returns null.
    /// @return PAPI counting set.
    static CountingSet* get()
    {
        static counter s_cnt;
        return s_cnt.set_;
    }

    //--------------------------------------------------------------------------
    /// Inserts element into the PAPI counting set.
    /// Without PAPI, does nothing.
    template <typename T>
    static void insert( T element, Id id )
    {
        #ifdef BLAS_HAVE_PAPI
            get()->insert( element, uint32_t( id ) );
        #endif
    }

    //--------------------------------------------------------------------------
    /// Inserts element with hashable_size into the PAPI counting set.
    /// hashable_size <= sizeof(element).
    /// Without PAPI, does nothing.
    template <typename T>
    static void insert( size_t hashable_size, T element, Id id )
    {
        #ifdef BLAS_HAVE_PAPI
            get()->insert( hashable_size, element, uint32_t( id ) );
        #endif
    }

    //--------------------------------------------------------------------------
    /// TODO
    /// Prints out all elements in the BLAS++ counting set.
    /// Without PAPI, does nothing.
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
                                op2char( ptr->trans ), llong( ptr->m ), llong( ptr->n ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemv: {
                        auto *ptr = static_cast<hemv_type *>( iter->ptr );
                        double flop = Gflop<double>::hemv( ptr->n ) * 1e9 * iter->count;
                        printf( "hemv( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symv: {
                        auto *ptr = static_cast<symv_type *>( iter->ptr );
                        double flop = Gflop<double>::symv( ptr->n ) * 1e9 * iter->count;
                        printf( "symv( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmv: {
                        auto *ptr = static_cast<trmv_type *>( iter->ptr );
                        double flop = Gflop<double>::trmv( ptr->n ) * 1e9 * iter->count;
                        printf( "trmv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                diag2char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsv: {
                        auto *ptr = static_cast<trsv_type *>( iter->ptr );
                        double flop = Gflop<double>::trsv( ptr->n ) * 1e9 * iter->count;
                        printf( "trsv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                diag2char( ptr->diag), llong( ptr->n ), iter->count, flop );
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
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2: {
                        auto *ptr = static_cast<her_type *>( iter->ptr );
                        double flop = Gflop<double>::her2( ptr->n ) * 1e9 * iter->count;
                        printf( "her2( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr: {
                        auto *ptr = static_cast<syr_type *>( iter->ptr );
                        double flop = Gflop<double>::syr( ptr->n ) * 1e9 * iter->count;
                        printf( "syr( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2: {
                        auto *ptr = static_cast<syr2_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2( ptr->n ) * 1e9 * iter->count;
                        printf( "syr2( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Level 3 BLAS
                    case Id::gemm: {
                        auto *ptr = static_cast<gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "gemm( %c, %c, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                op2char( ptr->transA ), op2char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemm: {
                        auto *ptr = static_cast<hemm_type *>( iter->ptr );
                        double flop = Gflop<double>::hemm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "hemm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2k: {
                        auto *ptr = static_cast<her2k_type *>( iter->ptr );
                        double flop = Gflop<double>::her2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "her2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::herk: {
                        auto *ptr = static_cast<herk_type *>( iter->ptr );
                        double flop = Gflop<double>::herk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "herk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symm: {
                        auto *ptr = static_cast<symm_type *>( iter->ptr );
                        double flop = Gflop<double>::symm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "symm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2k: {
                        auto *ptr = static_cast<syr2k_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "syr2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syrk: {
                        auto *ptr = static_cast<syrk_type *>( iter->ptr );
                        double flop = Gflop<double>::syrk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "syrk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmm: {
                        auto *ptr = static_cast<trmm_type *>( iter->ptr );
                        double flop = Gflop<double>::trmm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "trmm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsm: {
                        auto *ptr = static_cast<trsm_type *>( iter->ptr );
                        double flop = Gflop<double>::trsm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "trsm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Device BLAS
                    case Id::dev_copy: {
                        auto *ptr = static_cast<dev_copy_type *>( iter->ptr );
                        double flop = Gflop<double>::copy( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_copy( %lld ) count %d, flop count %.2e\n",
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
                    case Id::dev_gemm: {
                        auto *ptr = static_cast<dev_gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_gemm( %c, %c, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                op2char( ptr->transA ), op2char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_hemm: {
                        auto *ptr = static_cast<dev_hemm_type *>( iter->ptr );
                        double flop = Gflop<double>::hemm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_hemm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_her2k: {
                        auto *ptr = static_cast<dev_her2k_type *>( iter->ptr );
                        double flop = Gflop<double>::her2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_her2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_herk: {
                        auto *ptr = static_cast<dev_herk_type *>( iter->ptr );
                        double flop = Gflop<double>::herk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_herk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
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
                    case Id::dev_scal: {
                        auto *ptr = static_cast<dev_scal_type *>( iter->ptr );
                        double flop = Gflop<double>::scal( ptr->n ) * 1e9 * iter->count;
                        printf( "dev_scal( %lld ) count %d, flop count %.2e\n",
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
                    case Id::dev_symm: {
                        auto *ptr = static_cast<dev_symm_type *>( iter->ptr );
                        double flop = Gflop<double>::symm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_symm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syr2k: {
                        auto *ptr = static_cast<dev_syr2k_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2k( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_syr2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_syrk: {
                        auto *ptr = static_cast<dev_syrk_type *>( iter->ptr );
                        double flop = Gflop<double>::syrk( ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_syrk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trmm: {
                        auto *ptr = static_cast<dev_trmm_type *>( iter->ptr );
                        double flop = Gflop<double>::trmm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trmm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::dev_trsm: {
                        auto *ptr = static_cast<dev_trsm_type *>( iter->ptr );
                        double flop = Gflop<double>::trsm( ptr->side, ptr->m, ptr->n ) * 1e9 * iter->count;
                        printf( "dev_trsm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }

                    // Device batch BLAS
                    case Id::dev_batch_gemm: {
                        auto *ptr = static_cast<dev_batch_gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9 * iter->count;
                        printf( "dev_batch_gemm( %c, %c, %lld, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                op2char( ptr->transA ), op2char( ptr->transB ),
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
    /// Constructor initializes PAPI counting set on first call to get().
    counter():
        set_( nullptr )
    {
        #ifdef BLAS_HAVE_PAPI
            papi_sde::PapiSde sde( "blas" );
            set_ = sde.create_counting_set( "counter" );
        #endif
    }

    CountingSet* set_;
};  // class count

}  // namespace blas

#endif  // BLAS_COUNTER_HH
