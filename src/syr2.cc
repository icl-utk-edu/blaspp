#include "blas/fortran.h"
#include "blas.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    float alpha,
    float const *x, int64_t incx,
    float const *y, int64_t incy,
    float       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    BLAS_ssyr2( &uplo_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    double alpha,
    double const *x, int64_t incx,
    double const *y, int64_t incy,
    double       *A, int64_t lda )
{
    // check arguments
    blas_error_if( layout != Layout::ColMajor &&
                   layout != Layout::RowMajor );
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );
    blas_error_if( n < 0 );
    blas_error_if( lda < n );
    blas_error_if( incx == 0 );
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;
    blas_int incy_ = (blas_int) incy;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    char uplo_ = uplo2char( uplo );
    BLAS_dsyr2( &uplo_, &n_, &alpha, x, &incx_, y, &incy_, A, &lda_ );
}

// -----------------------------------------------------------------------------
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<float> alpha,
    std::complex<float> const *x, int64_t incx,
    std::complex<float> const *y, int64_t incy,
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
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    // if x2=x and y2=y, then they aren't modified
    std::complex<float> *x2 = const_cast< std::complex<float>* >( x );
    std::complex<float> *y2 = const_cast< std::complex<float>* >( y );

    // no [cz]syr2 in BLAS or LAPACK, so use [cz]syr2k with k=1 and beta=1.
    // if   inc == 1, consider x and y as n-by-1 matrices in n-by-1 arrays,
    // elif inc >= 1, consider x and y as 1-by-n matrices in inc-by-n arrays,
    // else, copy x and y and use case inc == 1 above.
    blas_int k_ = 1;
    char trans_;
    blas_int ldx_, ldy_;
    if (incx == 1 && incy == 1) {
        trans_ = 'N';
        ldx_ = n_;
        ldy_ = n_;
    }
    else if (incx >= 1 && incy >= 1) {
        trans_ = 'T';
        ldx_ = (blas_int) incx;
        ldy_ = (blas_int) incy;
    }
    else {
        x2 = new std::complex<float>[n];
        y2 = new std::complex<float>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            x2[i] = x[ix];
            y2[i] = y[iy];
            ix += incx;
            iy += incy;
        }
        trans_ = 'N';
        ldx_ = n_;
        ldy_ = n_;
    }
    std::complex<float> beta = 1;

    char uplo_ = uplo2char( uplo );
    BLAS_csyr2k( &uplo_, &trans_, &n_, &k_,
                 (blas_complex_float*) &alpha,
                 (blas_complex_float*) x2, &ldx_,
                 (blas_complex_float*) y2, &ldy_,
                 (blas_complex_float*) &beta,
                 (blas_complex_float*) A, &lda_ );

    if (x2 != x) {
        delete[] x2;
        delete[] y2;
    }
}

// -----------------------------------------------------------------------------
/// @ingroup syr2
void syr2(
    blas::Layout layout,
    blas::Uplo uplo,
    int64_t n,
    std::complex<double> alpha,
    std::complex<double> const *x, int64_t incx,
    std::complex<double> const *y, int64_t incy,
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
    blas_error_if( incy == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        blas_error_if( n              > std::numeric_limits<blas_int>::max() );
        blas_error_if( lda            > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incx) > std::numeric_limits<blas_int>::max() );
        blas_error_if( std::abs(incy) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
    }

    // if x2=x and y2=y, then they aren't modified
    std::complex<double> *x2 = const_cast< std::complex<double>* >( x );
    std::complex<double> *y2 = const_cast< std::complex<double>* >( y );

    // no [cz]syr2 in BLAS or LAPACK, so use [cz]syr2k with k=1 and beta=1.
    // if   inc == 1, consider x and y as n-by-1 matrices in n-by-1 arrays,
    // elif inc >= 1, consider x and y as 1-by-n matrices in inc-by-n arrays,
    // else, copy x and y and use case inc == 1 above.
    blas_int k_ = 1;
    char trans_;
    blas_int ldx_, ldy_;
    if (incx == 1 && incy == 1) {
        trans_ = 'N';
        ldx_ = n_;
        ldy_ = n_;
    }
    else if (incx >= 1 && incy >= 1) {
        trans_ = 'T';
        ldx_ = (blas_int) incx;
        ldy_ = (blas_int) incy;
    }
    else {
        x2 = new std::complex<double>[n];
        y2 = new std::complex<double>[n];
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        int64_t iy = (incy > 0 ? 0 : (-n + 1)*incy);
        for (int64_t i = 0; i < n; ++i) {
            x2[i] = x[ix];
            y2[i] = y[iy];
            ix += incx;
            iy += incy;
        }
        trans_ = 'N';
        ldx_ = n_;
        ldy_ = n_;
    }
    std::complex<double> beta = 1;

    char uplo_ = uplo2char( uplo );
    BLAS_zsyr2k( &uplo_, &trans_, &n_, &k_,
                 (blas_complex_double*) &alpha,
                 (blas_complex_double*) x2, &ldx_,
                 (blas_complex_double*) y2, &ldy_,
                 (blas_complex_double*) &beta,
                 (blas_complex_double*) A, &lda_ );

    if (x2 != x) {
        delete[] x2;
        delete[] y2;
    }
}

}  // namespace blas
