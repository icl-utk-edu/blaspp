// Copyright (c) 2017-2022, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas/device_blas.hh"

#include "device_internal.hh"

#include <limits>

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::trsm(
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
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if( ldda < m );
    else
        blas_error_if( ldda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( lddb < m );
    else
        blas_error_if( lddb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_    = (device_blas_int) m;
    device_blas_int n_    = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    blas::set_device( queue.device() );
    device::strsm(
            queue,
            side, uplo, trans, diag,
            m_, n_, alpha,
            dA, ldda_,
            dB, lddb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::trsm(
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
    blas::Queue  &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if( ldda < m );
    else
        blas_error_if( ldda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( lddb < m );
    else
        blas_error_if( lddb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_    = (device_blas_int) m;
    device_blas_int n_    = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    blas::set_device( queue.device() );
    device::dtrsm(
            queue,
            side, uplo, trans, diag,
            m_, n_, alpha,
            dA, ldda_,
            dB, lddb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::trsm(
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
    blas::Queue  &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if( ldda < m );
    else
        blas_error_if( ldda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( lddb < m );
    else
        blas_error_if( lddb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_    = (device_blas_int) m;
    device_blas_int n_    = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    blas::set_device( queue.device() );
    device::ctrsm(
            queue,
            side, uplo, trans, diag,
            m_, n_, alpha,
            dA, ldda_,
            dB, lddb_ );
}

// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::trsm(
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
    blas::Queue  &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if( ldda < m );
    else
        blas_error_if( ldda < n );

    if (layout == Layout::ColMajor)
        blas_error_if( lddb < m );
    else
        blas_error_if( lddb < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_    = (device_blas_int) m;
    device_blas_int n_    = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper, left <=> right, m <=> n
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        side = (side == Side::Left ? Side::Right : Side::Left);
        std::swap( m_, n_ );
    }

    blas::set_device( queue.device() );
    device::ztrsm(
            queue,
            side, uplo, trans, diag,
            m_, n_, alpha,
            dA, ldda_,
            dB, lddb_ );
}
