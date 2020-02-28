#include "blas/device_blas.hh"
#include <limits>

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup hemm
void blas::hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    float alpha,
    float const *dA, int64_t ldda,
    float const *dB, int64_t lddb,
    float beta,
    float       *dC, int64_t lddc,
    blas::Queue &queue )
{
    blas::symm( layout, side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue );
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
void blas::hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    double alpha,
    double const *dA, int64_t ldda,
    double const *dB, int64_t lddb,
    double beta,
    double       *dC, int64_t lddc,
    blas::Queue &queue )
{
    symm( layout, side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
void blas::hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *dA, int64_t ldda,
    std::complex<float> const *dB, int64_t lddb,
    std::complex<float> beta,
    std::complex<float>       *dC, int64_t lddc,
    blas::Queue &queue )
{
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( ldda < m, "ldda %lld < m %lld", (lld) ldda, (lld) m );
    else
        blas_error_if_msg( ldda < n, "ldda %lld < n %lld", (lld) ldda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( lddb < m );
        blas_error_if( lddc < m );
    }
    else {
        blas_error_if( lddb < n );
        blas_error_if( lddc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_   = (device_blas_int) m;
    device_blas_int n_   = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap left <=> right, lower <=> upper, m <=> n
        side = (side == Side::Left  ? Side::Right : Side::Left);
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        std::swap( m_, n_ );
    }

    device_side_t side_ = device_side_const( side );
    device_uplo_t uplo_ = device_uplo_const( uplo );

    blas::set_device( queue.device() );
    DEVICE_chemm(
            queue.handle(), side_, uplo_, m_, n_,
            alpha, dA, ldda_,
                   dB, lddb_,
            beta,  dC, lddc_ );
}

// -----------------------------------------------------------------------------
/// @ingroup hemm
void blas::hemm(
    blas::Layout layout,
    blas::Side side,
    blas::Uplo uplo,
    int64_t m, int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *dA, int64_t ldda,
    std::complex<double> const *dB, int64_t lddb,
    std::complex<double> beta,
    std::complex<double>       *dC, int64_t lddc,
    blas::Queue &queue )
{
    typedef long long lld;

    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( side != Side::Left &&
                   side != Side::Right );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( m < 0 );
    blas_error_if( n < 0 );

    if (side == Side::Left)
        blas_error_if_msg( ldda < m, "ldda %lld < m %lld", (lld) ldda, (lld) m );
    else
        blas_error_if_msg( ldda < n, "ldda %lld < n %lld", (lld) ldda, (lld) n );

    if (layout == Layout::ColMajor) {
        blas_error_if( lddb < m );
        blas_error_if( lddc < m );
    }
    else {
        blas_error_if( lddb < n );
        blas_error_if( lddc < n );
    }

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(device_blas_int)) {
        blas_error_if( m   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( n   > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( ldda > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddb > std::numeric_limits<device_blas_int>::max() );
        blas_error_if( lddc > std::numeric_limits<device_blas_int>::max() );
    }

    device_blas_int m_   = (device_blas_int) m;
    device_blas_int n_   = (device_blas_int) n;
    device_blas_int ldda_ = (device_blas_int) ldda;
    device_blas_int lddb_ = (device_blas_int) lddb;
    device_blas_int lddc_ = (device_blas_int) lddc;

    if (layout == Layout::RowMajor) {
        // swap left <=> right, lower <=> upper, m <=> n
        side = (side == Side::Left  ? Side::Right : Side::Left);
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        std::swap( m_, n_ );
    }

    device_side_t side_ = device_side_const( side );
    device_uplo_t uplo_ = device_uplo_const( uplo );

    blas::set_device( queue.device() );
    DEVICE_zhemm(
            queue.handle(), side_, uplo_, m_, n_,
            alpha, dA, ldda_,
                   dB, lddb_,
            beta,  dC, lddc_ );
}
