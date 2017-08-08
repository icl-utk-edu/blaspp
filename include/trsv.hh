#ifndef BLAS_TRSV_HH
#define BLAS_TRSV_HH

#include "blas_fortran.hh"
#include "blas_util.hh"

#include <limits>

namespace blas {

// =============================================================================
// Overloaded wrappers for s, d, c, z precisions.

// -----------------------------------------------------------------------------
inline
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    float const *A, int64_t lda,
    float       *x, int64_t incx )
{
    printf( "strsv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    f77_strsv( &uplo_, &trans_, &diag_, &n_, A, &lda_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    double const *A, int64_t lda,
    double       *x, int64_t incx )
{
    printf( "dtrsv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);
    }

    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    f77_dtrsv( &uplo_, &trans_, &diag_, &n_, A, &lda_, x, &incx_ );
}

// -----------------------------------------------------------------------------
inline
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<float> const *A, int64_t lda,
    std::complex<float>       *x, int64_t incx )
{
    printf( "ctrsv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);

        // conjugate x (in-place)
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x[ix] = conj( x[ix] );
            ix += incx;
        }
    }

    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    f77_ctrsv( &uplo_, &trans_, &diag_, &n_, A, &lda_, x, &incx_ );

    if (layout == Layout::RowMajor) {
        // conjugate x (in-place)
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x[ix] = conj( x[ix] );
            ix += incx;
        }
    }
}

// -----------------------------------------------------------------------------
inline
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    std::complex<double> const *A, int64_t lda,
    std::complex<double>       *x, int64_t incx )
{
    printf( "ztrsv implementation\n" );

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // check for overflow in native BLAS integer type, if smaller than int64_t
    if (sizeof(int64_t) > sizeof(blas_int)) {
        throw_if_( n              > std::numeric_limits<blas_int>::max() );
        throw_if_( lda            > std::numeric_limits<blas_int>::max() );
        throw_if_( std::abs(incx) > std::numeric_limits<blas_int>::max() );
    }

    blas_int n_    = (blas_int) n;
    blas_int lda_  = (blas_int) lda;
    blas_int incx_ = (blas_int) incx;

    if (layout == Layout::RowMajor) {
        // swap lower <=> upper
        // A => A^T; A^T => A; A^H => A
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        trans = (trans == Op::NoTrans ? Op::Trans : Op::NoTrans);

        // conjugate x (in-place)
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x[ix] = conj( x[ix] );
            ix += incx;
        }
    }

    char uplo_  = uplo2char( uplo );
    char trans_ = op2char( trans );
    char diag_  = diag2char( diag );
    f77_ztrsv( &uplo_, &trans_, &diag_, &n_, A, &lda_, x, &incx_ );

    if (layout == Layout::RowMajor) {
        // conjugate x (in-place)
        int64_t ix = (incx > 0 ? 0 : (-n + 1)*incx);
        for (int64_t i = 0; i < n; ++i) {
            x[ix] = conj( x[ix] );
            ix += incx;
        }
    }
}

// =============================================================================
/// Solve the triangular matrix-vector equation
///     op(A)*x = b,
/// where op(A) is one of
///     op(A) = A    or
///     op(A) = A^T  or
///     op(A) = A^H,
/// x and b are vectors, and A is an n-by-n, unit or non-unit,
/// upper or lower triangular matrix.
///
/// No test for singularity or near-singularity is included in this
/// routine. Such tests must be performed before calling this routine.
/// @see latrs for a more numerically robust implementation.
///
/// Generic implementation for arbitrary data types.
///
/// @param[in] layout
///         Matrix storage, Layout::ColMajor or Layout::RowMajor.
///
/// @param[in] uplo
///         TODO
///
/// @param[in] trans
///         The equation to be solved:
///         trans = Op::NoTrans   is A*x   = b,
///         trans = Op::Trans     is A^T*x = b,
///         trans = Op::ConjTrans is A^H*x = b.
///
/// @param[in] diag
///         TODO
///
/// @param[in] n
///         Number of rows and columns of the matrix A.
///
/// @param[in] A
///         The n-by-n matrix A, stored in an lda-by-n array.
///
/// @param[in] lda
///         Leading dimension of A, i.e., column stride. lda >= max(1,m).
///
/// @param[in,out] x
///         The n-element vector x, of length (n-1)*abs(incx) + 1.
///
/// @param[in] incx
///         Stride between elements of x. incx must not be zero.
///         If incx < 0, uses elements of x in reverse order: x(n-1), ..., x(0).
///
/// @ingroup blas2

template< typename TA, typename TX >
void trsv(
    blas::Layout layout,
    blas::Uplo uplo,
    blas::Op trans,
    blas::Diag diag,
    int64_t n,
    TA const *A, int64_t lda,
    TX       *x, int64_t incx )
{
    printf( "template trsv implementation\n" );

    typedef typename blas::traits2<TA, TX>::scalar_t scalar_t;

    #define A(i_, j_) A[ (i_) + (j_)*lda ]

    // constants
    const scalar_t zero = 0;
    const scalar_t one  = 1;

    // check arguments
    throw_if_( layout != Layout::ColMajor &&
               layout != Layout::RowMajor );
    throw_if_( uplo != Uplo::Lower &&
               uplo != Uplo::Upper );
    throw_if_( trans != Op::NoTrans &&
               trans != Op::Trans &&
               trans != Op::ConjTrans );
    throw_if_( diag != Diag::NonUnit &&
               diag != Diag::Unit );
    throw_if_( n < 0 );
    throw_if_( lda < n );
    throw_if_( incx == 0 );

    // quick return
    if (n == 0)
        return;

    // for row major, swap lower <=> upper and
    // A => A^T; A^T => A; A^H => A & conj
    bool doconj = false;
    if (layout == Layout::RowMajor) {
        uplo = (uplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
        if (trans == Op::NoTrans) {
            trans = Op::Trans;
        }
        else {
            if (trans == Op::ConjTrans) {
                doconj = true;
            }
            trans = Op::NoTrans;
        }
    }

    bool nonunit = (diag == Diag::NonUnit);
    int64_t kx = (incx > 0 ? 0 : (-n + 1)*incx);

    if (trans == Op::NoTrans && ! doconj) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    if (nonunit) {
                        x[j] /= A(j, j);
                    }
                    TX tmp = x[j];
                    for (int64_t i = j - 1; i >= 0; --i) {
                        x[i] -= tmp * A(i, j);
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx + (n - 1)*incx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= A(j, j);
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        x[ix] -= tmp * A(i, j);
                    }
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[j] /= A(j, j);
                    }
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        x[i] -= tmp * A(i, j);
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= A(j, j);
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        x[ix] -= tmp * A(i, j);
                    }
                    jx += incx;
                }
            }
        }
    }
    else if (trans == Op::NoTrans && doconj) {
        // Form x := A^{-1} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero, for consistent NAN handling
                    if (nonunit) {
                        x[j] /= conj( A(j, j) );
                    }
                    TX tmp = x[j];
                    for (int64_t i = j - 1; i >= 0; --i) {
                        x[i] -= tmp * conj( A(i, j) );
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx + (n - 1)*incx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= conj( A(j, j) );
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j - 1; i >= 0; --i) {
                        ix -= incx;
                        x[ix] -= tmp * conj( A(i, j) );
                    }
                    jx -= incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[j] /= conj( A(j, j) );
                    }
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        x[i] -= tmp * conj( A(i, j) );
                    }
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    // note: NOT skipping if x[j] is zero ...
                    if (nonunit) {
                        x[jx] /= conj( A(j, j) );
                    }
                    TX tmp = x[jx];
                    int64_t ix = jx;
                    for (int64_t i = j+1; i < n; ++i) {
                        ix += incx;
                        x[ix] -= tmp * conj( A(i, j) );
                    }
                    jx += incx;
                }
            }
        }
    }
    else if (trans == Op::Trans) {
        // Form  x := A^{-T} * x
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= A(i, j) * x[i];
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    int64_t ix = kx;
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= A(i, j) * x[ix];
                        ix += incx;
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        tmp -= A(i, j) * x[i];
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                int64_t jx = kx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    int64_t ix = kx;
                    TX tmp = x[jx];
                    for (int64_t i = n - 1; i >= j + 1; --i) {
                        tmp -= A(i, j) * x[ix];
                        ix -= incx;
                    }
                    if (nonunit) {
                        tmp /= A(j, j);
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
    }
    else {
        // Form x := A^{-H} * x
        // same code as above A^{-T} * x case, except add conj()
        if (uplo == Uplo::Upper) {
            // upper
            if (incx == 1) {
                // unit stride
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[j];
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= conj( A(i, j) ) * x[i];
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                int64_t jx = kx;
                for (int64_t j = 0; j < n; ++j) {
                    TX tmp = x[jx];
                    int64_t ix = kx;
                    for (int64_t i = 0; i <= j - 1; ++i) {
                        tmp -= conj( A(i, j) ) * x[ix];
                        ix += incx;
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
        else {
            // lower
            if (incx == 1) {
                // unit stride
                for (int64_t j = n - 1; j >= 0; --j) {
                    TX tmp = x[j];
                    for (int64_t i = j + 1; i < n; ++i) {
                        tmp -= conj( A(i, j) ) * x[i];
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[j] = tmp;
                }
            }
            else {
                // non-unit stride
                kx += (n - 1)*incx;
                int64_t jx = kx;
                for (int64_t j = n - 1; j >= 0; --j) {
                    int64_t ix = kx;
                    TX tmp = x[jx];
                    for (int64_t i = n - 1; i >= j + 1; --i) {
                        tmp -= conj( A(i, j) ) * x[ix];
                        ix -= incx;
                    }
                    if (nonunit) {
                        tmp /= conj( A(j, j) );
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            }
        }
    }

    #undef A
}

}  // namespace blas

#endif        //  #ifndef BLAS_TRSV_HH
