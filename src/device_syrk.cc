#include "blas/device_blas.hh"
#include <limits>

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup syrk
void blas::syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    float alpha,
    float const *dA, int64_t ldda,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldda < n );
    else
        blas_error_if( ldda < k );

    blas_error_if( lddc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_   = (device_blas_int) n;
    device_blas_int k_   = (device_blas_int) k;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // dA => dA^T; dA^T => dA; dA^H => dA
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    device_uplo_t uplo_ = device_uplo_const( uplo );
    device_trans_t trans_ = device_trans_const( trans );
    blas::set_device( queue.device() );
    DEVICE_ssyrk(
            queue.handle(),
            uplo_, trans_,
            n_, k_,
            alpha, dA, ldda_,
            beta,  dC, lddc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void blas::syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    double alpha,
    double const *dA, int64_t ldda,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldda < n );
    else
        blas_error_if( ldda < k );

    blas_error_if( lddc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_   = (device_blas_int) n;
    device_blas_int k_   = (device_blas_int) k;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // dA => dA^T; dA^T => dA; dA^H => dA
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    device_uplo_t uplo_ = device_uplo_const( uplo );
    device_trans_t trans_ = device_trans_const( trans );
    blas::set_device( queue.device() );
    DEVICE_dsyrk(
            queue.handle(),
            uplo_, trans_,
            n_, k_,
            alpha, dA, ldda_,
            beta,  dC, lddc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void blas::syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldda < n );
    else
        blas_error_if( ldda < k );

    blas_error_if( lddc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_   = (device_blas_int) n;
    device_blas_int k_   = (device_blas_int) k;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // dA => dA^T; dA^T => dA; dA^H => dA
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    device_uplo_t uplo_ = device_uplo_const( uplo );
    device_trans_t trans_ = device_trans_const( trans );
    blas::set_device( queue.device() );
    DEVICE_csyrk(
            queue.handle(),
            uplo_, trans_,
            n_, k_,
            alpha, dA, ldda_,
            beta,  dC, lddc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syrk
void blas::syrk(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    int64_t n, int64_t k,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans );
    blas_error_if( n < 0 );
    blas_error_if( k < 0 );

    if ((trans == Op::NoTrans) ^ (layout == Layout::RowMajor))
        blas_error_if( ldda < n );
    else
        blas_error_if( ldda < k );

    blas_error_if( lddc < n );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( k   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int n_   = (device_blas_int) n;
    device_blas_int k_   = (device_blas_int) k;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // dA => dA^T; dA^T => dA; dA^H => dA
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    device_uplo_t uplo_ = device_uplo_const( uplo );
    device_trans_t trans_ = device_trans_const( trans );
    blas::set_device( queue.device() );
    DEVICE_zsyrk(
            queue.handle(),
            uplo_, trans_,
            n_, k_,
            alpha, dA, ldda_,
            beta,  dC, lddc_ );
}
