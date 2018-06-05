#include "blas_fortran.hh"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float       *A, int64_t lda )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda );
}

// -----------------------------------------------------------------------------
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double       *A, int64_t lda )
{
    syr( layout, uplo, n, alpha, x, incx, A, lda );
}

// -----------------------------------------------------------------------------
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    // if x2=x, then it isn't modified
    std::complex<float> *x2 = const_cast< std::complex<float>* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate x (in x2)
        x2 = new std::complex<float>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x2[i] = conj( x[ix] );
            ix += incx;
        }
        incx_ = 1;
    }

    char uplo_ = uplo2char( uplo );
    BLAS_cher( &uplo_, &n_,
               &alpha,
               (blas_complex_float*) x2, &incx_,
               (blas_complex_float*) A, &lda_ );

    if (layout == Layout::RowMajor) {
        delete[] x2;
    }
}

// -----------------------------------------------------------------------------
/// @ingroup her
void her(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double>       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    // if x2=x, then it isn't modified
    std::complex<double> *x2 = const_cast< std::complex<double>* >( x );
    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);

        // conjugate x (in x2)
        x2 = new std::complex<double>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x2[i] = conj( x[ix] );
            ix += incx;
        }
        incx_ = 1;
    }

    char uplo_ = uplo2char( uplo );
    BLAS_zher( &uplo_, &n_,
               &alpha,
               (blas_complex_double*) x2, &incx_,
               (blas_complex_double*) A, &lda_ );

    if (layout == Layout::RowMajor) {
        delete[] x2;
    }
}

}  // namespace blas
