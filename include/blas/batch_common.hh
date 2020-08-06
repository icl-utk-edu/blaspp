// Copyright (c) 2017-2020, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifndef BLAS_BATCH_COMMON_HH
#define BLAS_BATCH_COMMON_HH

#include "blas/util.hh"
#include <vector>

namespace blas {
namespace batch {

#define INTERNAL_INFO_DEFAULT    (-1000)

template<typename T>
T extract(std::vector<T> const &ivector, const int64_t index)
{
    return (ivector.size() == 1) ? ivector[0] : ivector[index];
}

// -----------------------------------------------------------------------------
// batch gemm check
template<typename T>
void gemm_check(
        blas::Layout                 layout,
        std::vector<blas::Op> const &transA,
        std::vector<blas::Op> const &transB,
        std::vector<int64_t>  const &m,
        std::vector<int64_t>  const &n,
        std::vector<int64_t>  const &k,
        std::vector<T >       const &alpha,
        std::vector<T*>       const &A, std::vector<int64_t> const &lda,
        std::vector<T*>       const &B, std::vector<int64_t> const &ldb,
        std::vector<T >       const &beta,
        std::vector<T*>       const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (transA.size() != 1 && transA.size() != batchCount) );
    blas_error_if( (transB.size() != 1 && transB.size() != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if( (B.size() != 1 && B.size() < batchCount) );
    blas_error_if( (C.size() < batchCount) );

    blas_error_if( A.size() == 1 && (m.size() > 1 || k.size() > 1 || lda.size() > 1) );
    blas_error_if( B.size() == 1 && (k.size() > 1 || n.size() > 1 || ldb.size() > 1) );
    blas_error_if( C.size() == 1 &&
               (transA.size() > 1 || transB.size() > 1 ||
                m.size()      > 1 || n.size()      > 1 || k.size()   > 1 ||
                alpha.size()  > 1 || beta.size()   > 1 ||
                lda.size()    > 1 || ldb.size()    > 1 || ldc.size() > 1 ||
                A.size()      > 1 || B.size()      > 1
                )
             );

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Op transA_ = extract<Op>(transA, i);
        Op transB_ = extract<Op>(transB, i);

        int64_t m_ = extract<int64_t>(m, i);
        int64_t n_ = extract<int64_t>(n, i);
        int64_t k_ = extract<int64_t>(k, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = ((transA_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? m_ : k_;
        int64_t nrowB_ = ((transB_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? k_ : n_;
        int64_t nrowC_ = (layout == Layout::ColMajor) ? k_ : n_;

        internal_info[i] = 0;
        if (transA_ != Op::NoTrans &&
           transA_ != Op::Trans   &&
           transA_ != Op::ConjTrans) {
            internal_info[i] = -2;
        }
        else if (transB_ != Op::NoTrans &&
                transB_ != Op::Trans   &&
                transB_ != Op::ConjTrans) {
            internal_info[i] = -3;
        }
        else if (m_ < 0) internal_info[i] = -4;
        else if (n_ < 0) internal_info[i] = -5;
        else if (k_ < 0) internal_info[i] = -6;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldb_ < nrowB_) internal_info[i] = -11;
        else if (ldc_ < nrowC_) internal_info[i] = -14;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
        blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
        blas_error_if_msg( info_ != 0, "One or more non-zero entry in vector info");
    }
}

// -----------------------------------------------------------------------------
// batch trsm check
template<typename T>
void trsm_check(
        blas::Layout                   layout,
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m,
        std::vector<int64_t>    const &n,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );
    blas_error_if( (diag.size()  != 1 && diag.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if(  B.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );

    blas_error_if( A.size() == 1 && ( lda.size()  > 1                         ||
                                      side.size() > 1                         ||
                                     (side[0] == Side::Left  && m.size() > 1) ||
                                     (side[0] == Side::Right && n.size() > 1) ));
    blas_error_if( B.size() == 1 && ( side.size()  > 1 || uplo.size() > 1 ||
                                      trans.size() > 1 || diag.size() > 1 ||
                                      m.size()     > 1 || n.size()    > 1 ||
                                      alpha.size() > 1 || A.size()    > 1 ||
                                      lda.size()   > 1 || ldb.size()  > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Side  side_ = extract<Side>(side , i);
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op  >(trans, i);
        Diag  diag_ = extract<Diag>(diag , i);

        int64_t m_ = extract<int64_t>(m, i);
        int64_t n_ = extract<int64_t>(n, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);

        int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;
        int64_t nrowB_ = (layout == Layout::ColMajor) ? m_ : n_;

        internal_info[i] = 0;
        if (side_ != Side::Left && side_ != Side::Right) {
            internal_info[i] = -2;
        }
        else if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -3;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::Trans && trans_ != Op::ConjTrans) {
            internal_info[i] = -4;
        }
        else if (diag_ != Diag::NonUnit && diag_ != Diag::Unit) {
            internal_info[i] = -5;
        }
        else if (m_ < 0) internal_info[i] = -6;
        else if (n_ < 0) internal_info[i] = -7;
        else if (lda_ < nrowA_) internal_info[i] = -10;
        else if (ldb_ < nrowB_) internal_info[i] = -12;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch trmm check
template<typename T>
void trmm_check(
        blas::Layout                   layout,
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m,
        std::vector<int64_t>    const &n,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );
    blas_error_if( (diag.size()  != 1 && diag.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if(  B.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );

    blas_error_if( A.size() == 1 && ( lda.size()  > 1                         ||
                                      side.size() > 1                         ||
                                     (side[0] == Side::Left  && m.size() > 1) ||
                                     (side[0] == Side::Right && n.size() > 1) ));
    blas_error_if( B.size() == 1 && ( side.size()  > 1 || uplo.size() > 1 ||
                                      trans.size() > 1 || diag.size() > 1 ||
                                      m.size()     > 1 || n.size()    > 1 ||
                                      alpha.size() > 1 || A.size()    > 1 ||
                                      lda.size()   > 1 || ldb.size()  > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Side  side_ = extract<Side>(side , i);
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op  >(trans, i);
        Diag  diag_ = extract<Diag>(diag , i);

        int64_t m_ = extract<int64_t>(m, i);
        int64_t n_ = extract<int64_t>(n, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);

        int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;
        int64_t nrowB_ = (layout == Layout::ColMajor) ? m_ : n_;

        internal_info[i] = 0;
        if (side_ != Side::Left && side_ != Side::Right) {
            internal_info[i] = -2;
        }
        else if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -3;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::Trans && trans_ != Op::ConjTrans) {
            internal_info[i] = -4;
        }
        else if (diag_ != Diag::NonUnit && diag_ != Diag::Unit) {
            internal_info[i] = -5;
        }
        else if (m_ < 0) internal_info[i] = -6;
        else if (n_ < 0) internal_info[i] = -7;
        else if (lda_ < nrowA_) internal_info[i] = -10;
        else if (ldb_ < nrowB_) internal_info[i] = -12;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
        blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
        blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch hemm check
template<typename T>
void hemm_check(
        blas::Layout                   layout,
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<int64_t>    const &m,
        std::vector<int64_t>    const &n,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        std::vector<T>          const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (side.size()  != 1 && side.size()  != batchCount) );
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );

    blas_error_if( (m.size() != 1 && m.size() != batchCount) );
    blas_error_if( (n.size() != 1 && n.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if( (B.size() != 1 && B.size() < batchCount) );
    blas_error_if(  C.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 &&
                  (lda.size()  > 1                          ||
                   side.size() > 1                          ||
                   (side[0] == Side::Left  && m.size() > 1) ||
                   (side[0] == Side::Right && n.size() > 1) ));

    blas_error_if( B.size() == 1 &&
                  (m.size()   > 1 ||
                   n.size()   > 1 ||
                   ldb.size() > 1 ));

    blas_error_if( C.size() == 1 &&
                  (side.size()  > 1 ||
                   uplo.size()  > 1 ||
                   m.size()     > 1 ||
                   n.size()     > 1 ||
                   alpha.size() > 1 ||
                   A.size()     > 1 ||
                   lda.size()   > 1 ||
                   B.size()     > 1 ||
                   ldb.size()   > 1 ||
                   beta.size()  > 1 ||
                   ldc.size()   > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Side  side_ = extract<Side>(side , i);
        Uplo  uplo_ = extract<Uplo>(uplo , i);

        int64_t m_ = extract<int64_t>(m, i);
        int64_t n_ = extract<int64_t>(n, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = (side_ == Side::Left) ? m_ : n_;
        int64_t nrowB_ = (layout == Layout::ColMajor) ? m_ : n_;
        int64_t nrowC_ = (layout == Layout::ColMajor) ? m_ : n_;

        internal_info[i] = 0;
        if (side_ != Side::Left && side_ != Side::Right) {
            internal_info[i] = -2;
        }
        else if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -3;
        }
        else if (m_ < 0) internal_info[i] = -4;
        else if (n_ < 0) internal_info[i] = -5;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldb_ < nrowB_) internal_info[i] = -10;
        else if (ldc_ < nrowC_) internal_info[i] = -13;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch herk check
template<typename T, typename scalarT>
void herk_check(
        blas::Layout                   layout,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<int64_t>    const &n,
        std::vector<int64_t>    const &k,
        std::vector<scalarT>    const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<scalarT>    const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if(  C.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 &&
                  (lda.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 &&
                  (uplo.size()  > 1 ||
                   trans.size() > 1 ||
                   n.size()     > 1 ||
                   k.size()     > 1 ||
                   alpha.size() > 1 ||
                   A.size()     > 1 ||
                   lda.size()   > 1 ||
                   beta.size()  > 1 ||
                   ldc.size()   > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op>(trans , i);

        int64_t n_ = extract<int64_t>(n, i);
        int64_t k_ = extract<int64_t>(k, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;

        internal_info[i] = 0;
        if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -2;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::ConjTrans) {
            internal_info[i] = -3;
        }
        else if (n_ < 0) internal_info[i] = -4;
        else if (k_ < 0) internal_info[i] = -5;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldc_ < n_) internal_info[i] = -11;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch hemm check
template<typename T>
void symm_check(
        blas::Layout                   layout,
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<int64_t>    const &m,
        std::vector<int64_t>    const &n,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        std::vector<T>          const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    hemm_check(layout, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc, batchCount, info);
}

// -----------------------------------------------------------------------------
// batch syrk check
template<typename T>
void syrk_check(
        blas::Layout                   layout,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<int64_t>    const &n,
        std::vector<int64_t>    const &k,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T>          const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if(  C.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 &&
                  (lda.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 &&
                  (uplo.size()  > 1 ||
                   trans.size() > 1 ||
                   n.size()     > 1 ||
                   k.size()     > 1 ||
                   alpha.size() > 1 ||
                   A.size()     > 1 ||
                   lda.size()   > 1 ||
                   beta.size()  > 1 ||
                   ldc.size()   > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op>(trans , i);

        int64_t n_ = extract<int64_t>(n, i);
        int64_t k_ = extract<int64_t>(k, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;

        internal_info[i] = 0;
        if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -2;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::Trans) {
            internal_info[i] = -3;
        }
        else if (n_ < 0) internal_info[i] = -4;
        else if (k_ < 0) internal_info[i] = -5;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldc_ < n_) internal_info[i] = -11;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch her2k check
template<typename T, typename scalarT>
void her2k_check(
        blas::Layout                   layout,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<int64_t>    const &n,
        std::vector<int64_t>    const &k,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        std::vector<scalarT>    const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if( (B.size() != 1 && B.size() < batchCount) );
    blas_error_if(  C.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 &&
                  (lda.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( B.size() == 1 &&
                  (ldb.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 &&
                  (uplo.size()  > 1 ||
                   trans.size() > 1 ||
                   n.size()     > 1 ||
                   k.size()     > 1 ||
                   alpha.size() > 1 ||
                   A.size()     > 1 ||
                   lda.size()   > 1 ||
                   B.size()     > 1 ||
                   ldb.size()   > 1 ||
                   beta.size()  > 1 ||
                   ldc.size()   > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op>(trans , i);

        int64_t n_ = extract<int64_t>(n, i);
        int64_t k_ = extract<int64_t>(k, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;
        int64_t nrowB_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;

        internal_info[i] = 0;
        if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -2;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::ConjTrans) {
            internal_info[i] = -3;
        }
        else if (n_ < 0) internal_info[i] = -4;
        else if (k_ < 0) internal_info[i] = -5;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldb_ < nrowB_) internal_info[i] = -10;
        else if (ldc_ < n_) internal_info[i] = -13;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

// -----------------------------------------------------------------------------
// batch syr2k check
template<typename T>
void syr2k_check(
        blas::Layout                   layout,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<int64_t>    const &n,
        std::vector<int64_t>    const &k,
        std::vector<T>          const &alpha,
        std::vector<T*>         const &A, std::vector<int64_t> const &lda,
        std::vector<T*>         const &B, std::vector<int64_t> const &ldb,
        std::vector<T>          const &beta,
        std::vector<T*>         const &C, std::vector<int64_t> const &ldc,
        const size_t batchCount, std::vector<int64_t> &info)
{
    // size error checking
    blas_error_if( (uplo.size()  != 1 && uplo.size()  != batchCount) );
    blas_error_if( (trans.size() != 1 && trans.size() != batchCount) );

    blas_error_if( (n.size() != 1 && n.size() != batchCount) );
    blas_error_if( (k.size() != 1 && k.size() != batchCount) );

    // to support checking errors for the group interface, batchCount will be equal to group_count
    // but the data arrays are generally >= group_count
    blas_error_if( (A.size() != 1 && A.size() < batchCount) );
    blas_error_if( (B.size() != 1 && B.size() < batchCount) );
    blas_error_if(  C.size() < batchCount );

    blas_error_if( (lda.size() != 1 && lda.size() != batchCount) );
    blas_error_if( (ldb.size() != 1 && ldb.size() != batchCount) );
    blas_error_if( (ldc.size() != 1 && ldc.size() != batchCount) );

    blas_error_if( (alpha.size() != 1 && alpha.size() != batchCount) );
    blas_error_if( (beta.size()  != 1 && beta.size()  != batchCount) );

    blas_error_if( A.size() == 1 &&
                  (lda.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( B.size() == 1 &&
                  (ldb.size()    > 1                  ||
                   n.size()      > 1                  ||
                   k.size()      > 1                  ||
                   (trans.size() > 1 && n[0] != k[0]) ));

    blas_error_if( C.size() == 1 &&
                  (uplo.size()  > 1 ||
                   trans.size() > 1 ||
                   n.size()     > 1 ||
                   k.size()     > 1 ||
                   alpha.size() > 1 ||
                   A.size()     > 1 ||
                   lda.size()   > 1 ||
                   B.size()     > 1 ||
                   ldb.size()   > 1 ||
                   beta.size()  > 1 ||
                   ldc.size()   > 1 ));

    int64_t* internal_info;
    if (info.size() == 1) {
        internal_info = new int64_t[batchCount];
    }
    else {
        internal_info = &info[0];
    }

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < batchCount; ++i) {
        Uplo  uplo_ = extract<Uplo>(uplo , i);
        Op   trans_ = extract<Op>(trans , i);

        int64_t n_ = extract<int64_t>(n, i);
        int64_t k_ = extract<int64_t>(k, i);

        int64_t lda_ = extract<int64_t>(lda, i);
        int64_t ldb_ = extract<int64_t>(ldb, i);
        int64_t ldc_ = extract<int64_t>(ldc, i);

        int64_t nrowA_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;
        int64_t nrowB_ = ((trans_ == Op::NoTrans) ^ (layout == Layout::RowMajor)) ? n_ : k_;

        internal_info[i] = 0;
        if (uplo_ != Uplo::Lower && uplo_ != Uplo::Upper) {
            internal_info[i] = -2;
        }
        else if (trans_ != Op::NoTrans && trans_ != Op::Trans) {
            internal_info[i] = -3;
        }
        else if (n_ < 0) internal_info[i] = -4;
        else if (k_ < 0) internal_info[i] = -5;
        else if (lda_ < nrowA_) internal_info[i] = -8;
        else if (ldb_ < nrowB_) internal_info[i] = -10;
        else if (ldc_ < n_) internal_info[i] = -13;
    }

    if (info.size() == 1) {
        // do a reduction that finds the first argument to encounter an error
        int64_t lerror = INTERNAL_INFO_DEFAULT;
        #pragma omp parallel for reduction(max:lerror)
        for (size_t i = 0; i < batchCount; ++i) {
            if (internal_info[i] == 0)
                continue;    // skip problems that passed error checks
            lerror = std::max(lerror, internal_info[i]);
        }
        info[0] = (lerror == INTERNAL_INFO_DEFAULT) ? 0 : lerror;

        // delete the internal vector
        delete[] internal_info;

        // throw an exception if needed
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
    else {
        int64_t info_ = 0;
        #pragma omp parallel for reduction(+:info_)
        for (size_t i = 0; i < batchCount; ++i) {
            info_ += info[i];
        }
         blas_error_if_msg( info[0] != 0, "info = %lld", (long long) info[0] );
    }
}

}  // namespace batch
}  // namespace blas

#endif        //  #ifndef BLAS_BATCH_COMMON_HH
