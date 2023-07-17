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
        gemm,
        gemv,
        ger,
        gerc,
        geru,
        hemm,
        hemv,
        her,
        her2,
        her2k,
        herk,
        symm,
        symv,
        syr,
        syr2,
        syr2k,
        syrk,
        trmm,
        trmv,
        trsm,
        trsv,
        // Add alphabetically.
    };

    //------------------------------------------------------------------------------
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

    //------------------------------------------------------------------------------
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
                    case Id::gemm: {
                        auto *ptr = static_cast<gemm_type *>( iter->ptr );
                        double flop = Gflop<double>::gemm( ptr->m, ptr->n, ptr->k ) * 1e9;
                        printf( "gemm( %c, %c, %lld, %lld, %lld ) count %d, flop count %.2e\n",
                                op2char( ptr->transA ), op2char( ptr->transB ),
                                llong( ptr->m ), llong( ptr->n ), llong( ptr->k ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemm: {
                        auto *ptr = static_cast<hemm_type *>( iter->ptr );
                        double flop = Gflop<double>::hemm( ptr->side, ptr->m, ptr->n ) * 1e9;
                        printf( "hemm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2k: {
                        auto *ptr = static_cast<her2k_type *>( iter->ptr );
                        double flop = Gflop<double>::her2k( ptr->n, ptr->k ) * 1e9;
                        printf( "her2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::herk: {
                        auto *ptr = static_cast<herk_type *>( iter->ptr );
                        double flop = Gflop<double>::herk( ptr->n, ptr->k ) * 1e9;
                        printf( "herk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symm: {
                        auto *ptr = static_cast<symm_type *>( iter->ptr );
                        double flop = Gflop<double>::symm( ptr->side, ptr->m, ptr->n ) * 1e9;
                        printf( "symm( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2k: {
                        auto *ptr = static_cast<syr2k_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2k( ptr->n, ptr->k ) * 1e9;
                        printf( "syr2k( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syrk: {
                        auto *ptr = static_cast<syrk_type *>( iter->ptr );
                        double flop = Gflop<double>::syrk( ptr->n, ptr->k ) * 1e9;
                        printf( "syrk( %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                llong( ptr->n ), llong( ptr->k ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmm: {
                        auto *ptr = static_cast<trmm_type *>( iter->ptr );
                        double flop = Gflop<double>::trmm( ptr->side, ptr->m, ptr->n ) * 1e9;
                        printf( "trmm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsm: {
                        auto *ptr = static_cast<trsm_type *>( iter->ptr );
                        double flop = Gflop<double>::trsm( ptr->side, ptr->m, ptr->n ) * 1e9;
                        printf( "trsm( %c, %c, %c, %c, %lld, %lld ) count %d, flop count %.2e\n",
                                side2char( ptr->side ), uplo2char( ptr->uplo ),
                                op2char( ptr->transA ), diag2char( ptr->diag ),
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::gemv: {
                        auto *ptr = static_cast<gemv_type *>( iter->ptr );
                        double flop = Gflop<double>::gemv( ptr->m, ptr->n ) * 1e9;
                        printf( "gemv( %c, %lld, %lld ) count %d, flop count %.2e\n",
                                op2char( ptr->trans ), llong( ptr->m ), llong( ptr->n ),
                                iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::hemv: {
                        auto *ptr = static_cast<hemv_type *>( iter->ptr );
                        double flop = Gflop<double>::hemv( ptr->n ) * 1e9;
                        printf( "hemv( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::symv: {
                        auto *ptr = static_cast<symv_type *>( iter->ptr );
                        double flop = Gflop<double>::symv( ptr->n ) * 1e9;
                        printf( "symv( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trmv: {
                        auto *ptr = static_cast<trmv_type *>( iter->ptr );
                        double flop = Gflop<double>::trmv( ptr->n ) * 1e9;
                        printf( "trmv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                diag2char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::trsv: {
                        auto *ptr = static_cast<trsv_type *>( iter->ptr );
                        double flop = Gflop<double>::trsv( ptr->n ) * 1e9;
                        printf( "trsv( %c, %c, %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ), op2char( ptr->trans ),
                                diag2char( ptr->diag), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::ger: {
                        auto *ptr = static_cast<ger_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9;
                        printf( "ger( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::geru: {
                        auto *ptr = static_cast<geru_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9;
                        printf( "geru (%lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::gerc: {
                        auto *ptr = static_cast<gerc_type *>( iter->ptr );
                        double flop = Gflop<double>::ger( ptr->m, ptr->n ) * 1e9;
                        printf( "gerc( %lld, %lld ) count %d, flop count %.2e\n",
                                llong( ptr->m ), llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her: {
                        auto *ptr = static_cast<her_type *>( iter->ptr );
                        double flop = Gflop<double>::her( ptr->n ) * 1e9;
                        printf( "her( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::her2: {
                        auto *ptr = static_cast<her_type *>( iter->ptr );
                        double flop = Gflop<double>::her2( ptr->n ) * 1e9;
                        printf( "her2( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr: {
                        auto *ptr = static_cast<syr_type *>( iter->ptr );
                        double flop = Gflop<double>::syr( ptr->n ) * 1e9;
                        printf( "syr( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
                        break;
                    }
                    case Id::syr2: {
                        auto *ptr = static_cast<syr2_type *>( iter->ptr );
                        double flop = Gflop<double>::syr2( ptr->n ) * 1e9;
                        printf( "syr2( %c, %lld ) count %d, flop count %.2e\n",
                                uplo2char( ptr->uplo ),llong( ptr->n ), iter->count, flop );
                        totalflops += flop;
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
