#ifndef DEVICE_BLAS_GEMM_HH
#define DEVICE_BLAS_GEMM_HH

#include "device.hh"
#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
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
    blas::Queue &queue )
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

    if ((transA == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldda < m );
    else
        blas_error_if( ldda < k );

    if ((transB == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( lddb < k );
    else
        blas_error_if( lddb < n );

    if (layout == Layout::ColMajor)
        blas_error_if( lddc < m );
    else
        blas_error_if( lddc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_trans_t  transA_ = blas::device_trans_const( transA );
    device_trans_t  transB_ = blas::device_trans_const( transB );
    device_blas_int m_      = (device_blas_int) m;
    device_blas_int n_      = (device_blas_int) n;
    device_blas_int k_      = (device_blas_int) k;
    device_blas_int ldda_   = (device_blas_int) ldda;
    device_blas_int lddb_   = (device_blas_int) lddb;
    device_blas_int lddc_   = (device_blas_int) lddc;
    
    blas::set_device( queue.device() );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        DEVICE_BLAS_sgemm( 
                queue.handle(), transB_, transA_, 
                n_, m_, k_, 
                alpha, dB, lddb_, dA, ldda_, 
                beta,  dC, lddc_);
    }
    else {
        DEVICE_BLAS_sgemm( 
                queue.handle(), transA_, transB_, 
                m_, n_, k_, 
                alpha, dA, ldda_, dB, lddb_, 
                beta,  dC, lddc_);
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
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
    blas::Queue &queue )
{
    /*
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
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB == Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB != Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }
    */

    device_trans_t  transA_ = blas::device_trans_const( transA );
    device_trans_t  transB_ = blas::device_trans_const( transB );
    device_blas_int m_      = (device_blas_int) m;
    device_blas_int n_      = (device_blas_int) n;
    device_blas_int k_      = (device_blas_int) k;
    device_blas_int ldda_   = (device_blas_int) ldda;
    device_blas_int lddb_   = (device_blas_int) lddb;
    device_blas_int lddc_   = (device_blas_int) lddc;

    blas::set_device( queue.device() );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        DEVICE_BLAS_dgemm( 
                queue.handle(), transB_, transA_, 
                n_, m_, k_, 
                alpha, dB, lddb_, dA, ldda_, 
                beta,  dC, lddc_);
    }
    else {
        DEVICE_BLAS_dgemm( 
                queue.handle(), transA_, transB_, 
                m_, n_, k_, 
                alpha, dA, ldda_, dB, lddb_, 
                beta,  dC, lddc_);
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
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
    blas::Queue &queue )
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
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB == Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB != Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_trans_t  transA_ = blas::device_trans_const( transA );
    device_trans_t  transB_ = blas::device_trans_const( transB );
    device_blas_int m_      = (device_blas_int) m;
    device_blas_int n_      = (device_blas_int) n;
    device_blas_int k_      = (device_blas_int) k;
    device_blas_int ldda_   = (device_blas_int) ldda;
    device_blas_int lddb_   = (device_blas_int) lddb;
    device_blas_int lddc_   = (device_blas_int) lddc;

    blas::set_device( queue.device() );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        DEVICE_BLAS_cgemm( 
                queue.handle(), transB_, transA_, 
                n_, m_, k_, 
                alpha, dB, lddb_, dA, ldda_, 
                beta,  dC, lddc_);
    }
    else {
        DEVICE_BLAS_cgemm( 
                queue.handle(), transA_, transB_, 
                m_, n_, k_, 
                alpha, dA, ldda_, dB, lddb_, 
                beta,  dC, lddc_);
    }
}

// -----------------------------------------------------------------------------
/// @ingroup gemm
inline
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
    blas::Queue &queue )
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
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB == Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < m );
    }
    else {
        if (transA != Op::NoTrans)
            blas_error_if( ldda < m );
        else
            blas_error_if( ldda < k );

        if (transB != Op::NoTrans)
            blas_error_if( lddb < k );
        else
            blas_error_if( lddb < n );

        blas_error_if( lddc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k    > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_trans_t  transA_ = blas::device_trans_const( transA );
    device_trans_t  transB_ = blas::device_trans_const( transB );
    device_blas_int m_      = (device_blas_int) m;
    device_blas_int n_      = (device_blas_int) n;
    device_blas_int k_      = (device_blas_int) k;
    device_blas_int ldda_   = (device_blas_int) ldda;
    device_blas_int lddb_   = (device_blas_int) lddb;
    device_blas_int lddc_   = (device_blas_int) lddc;

    blas::set_device( queue.device() );
    if (layout == Layout::RowMajor) {
        // swap transA <=> transB, m <=> n, B <=> A
        DEVICE_BLAS_zgemm( 
                queue.handle(), transB_, transA_, 
                n_, m_, k_, 
                alpha, dB, lddb_, dA, ldda_, 
                beta,  dC, lddc_);
    }
    else {
        DEVICE_BLAS_zgemm( 
                queue.handle(), transA_, transB_, 
                m_, n_, k_, 
                alpha, dA, ldda_, dB, lddb_, 
                beta,  dC, lddc_);
    }
}


}        //  namespace blas

#endif        //  #ifndef DEVICE_BLAS_GEMM_HH

