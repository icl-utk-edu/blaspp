#ifndef FLOPS_HH
#define FLOPS_HH

#include "blas.hh"

namespace blas {

// =============================================================================
// Level 1 BLAS

// -----------------------------------------------------------------------------
inline double fmuls_asum( double n )
    { return 0; }

inline double fadds_asum( double n )
    { return n-1; }

template< typename T >
inline double gflop_asum( double n, T* x )
{
    return (fmuls_asum(n) + fadds_asum(n)) / 1e9;
}

// todo: abs1 incurs adds, too
template< typename T >
inline double gflop_asum( double n, std::complex<T>* x )
{
    return (6*fmuls_asum(n) + 2*fadds_asum(n)) / 1e9;
}

template< typename T >
inline double gbyte_asum( double n, T* x )
{
    // read x
    return n * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_axpy( double n )
    { return n; }

inline double fadds_axpy( double n )
    { return n; }

template< typename T >
inline double gflop_axpy( double n, T* x )
{
    return (fmuls_axpy(n) + fadds_axpy(n)) / 1e9;
}

template< typename T >
inline double gflop_axpy( double n, std::complex<T>* x )
{
    return (6*fmuls_axpy(n) + 2*fadds_axpy(n)) / 1e9;
}

template< typename T >
inline double gbyte_axpy( double n, T* x )
{
    // read x, y; write y
    return 3*n * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_copy( double n, T* x )
{
    return 0;
}

template< typename T >
inline double gbyte_copy( double n, T* x )
{
    // read x; write y
    return 2*n * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_iamax( double n )
    { return 0; }

// n-1 compares, which are essentially adds (x > y is x - y > 0)
inline double fadds_iamax( double n )
    { return n-1; }

template< typename T >
inline double gflop_iamax( double n, T* x )
{
    return (fmuls_iamax(n) + fadds_iamax(n)) / 1e9;
}

// todo: abs1 incurs adds, too
template< typename T >
inline double gflop_iamax( double n, std::complex<T>* x )
{
    return (6*fmuls_iamax(n) + 2*fadds_iamax(n)) / 1e9;
}

template< typename T >
inline double gbyte_iamax( double n, T* x )
{
    // read x
    return n * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_nrm2( double n )
    { return n; }

inline double fadds_nrm2( double n )
    { return n-1; }

template< typename T >
inline double gflop_nrm2( double n, T* x )
{
    return (fmuls_nrm2(n) + fadds_nrm2(n)) / 1e9;
}

// todo: r*r + i*i, the r*i terms cancel
template< typename T >
inline double gflop_nrm2( double n, std::complex<T>* x )
{
    return (6*fmuls_nrm2(n) + 2*fadds_nrm2(n)) / 1e9;
}

template< typename T >
inline double gbyte_nrm2( double n, T* x )
{
    // read x
    return n * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_dot( double n )
    { return n; }

inline double fadds_dot( double n )
    { return n-1; }

template< typename T >
inline double gflop_dot( double n, T* x )
{
    return (fmuls_dot(n) + fadds_dot(n)) / 1e9;
}

template< typename T >
inline double gflop_dot( double n, std::complex<T>* x )
{
    return (6*fmuls_dot(n) + 2*fadds_dot(n)) / 1e9;
}

template< typename T >
inline double gbyte_dot( double n, T* x )
{
    // read x, y
    return 2*n * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_scal( double n )
    { return n; }

inline double fadds_scal( double n )
    { return 0; }

template< typename T >
inline double gflop_scal( double n, T* x )
{
    return (fmuls_scal(n) + fadds_scal(n)) / 1e9;
}

template< typename T >
inline double gflop_scal( double n, std::complex<T>* x )
{
    return (6*fmuls_scal(n) + 2*fadds_scal(n)) / 1e9;
}

template< typename T >
inline double gbyte_scal( double n, T* x )
{
    // read x; write x
    return 2*n * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_swap( double n, T* x )
{
    return 0;
}

template< typename T >
inline double gbyte_swap( double n, T* x )
{
    // read x, y; write x, y
    return 4*n * sizeof(T);
}


// =============================================================================
// Level 2 BLAS
// most formulas assume alpha=1, beta=0 or 1; otherwise add lower-order terms.
// i.e., this is minimum flops and bandwidth that could be consumed.

// -----------------------------------------------------------------------------
inline double fmuls_gemv( double m, double n )
    { return m*n; }

inline double fadds_gemv( double m, double n )
    { return m*n; }

template< typename T >
inline double gflop_gemv( double m, double n, T* x )
{
    return (fmuls_gemv(m, n) + fadds_gemv(m, n)) / 1e9;
}

template< typename T >
inline double gflop_gemv( double m, double n, std::complex<T>* x )
{
    return (6*fmuls_gemv(m, n) + 2*fadds_gemv(m, n)) / 1e9;
}

template< typename T >
inline double gbyte_gemv( double m, double n, T* x )
{
    // read A, x; write y
    return (m*n + m + n) * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_hemv( double n, T* x )
{
    return gflop_gemv( n, n, x );
}

template< typename T >
inline double gbyte_hemv( double n, T* x )
{
    // read A triangle, x; write y
    return (0.5*(n+1)*n + 2*n) * sizeof(T);
}

template< typename T >
inline double gflop_symv( double n, T* x )
{
    return gflop_hemv( n, x );
}

template< typename T >
inline double gbyte_symv( double n, T* x )
{
    return gbyte_hemv( n, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_trmv( double n )
    { return 0.5*n*(n + 1); }

inline double fadds_trmv( double n )
    { return 0.5*n*(n - 1); }

template< typename T >
inline double gflop_trmv( double n, T* x )
{
    return (fmuls_trmv(n) + fadds_trmv(n)) / 1e9;
}

template< typename T >
inline double gflop_trmv( double n, std::complex<T>* x )
{
    return (6*fmuls_trmv(n) + 2*fadds_trmv(n)) / 1e9;
}

template< typename T >
inline double gbyte_trmv( double n, T* x )
{
    // read A triangle, x; write x
    return (0.5*(n+1)*n + 2*n) * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_trsv( double n, T* x )
{
    return gflop_trmv( n, x );
}

template< typename T >
inline double gbyte_trsv( double n, T* x )
{
    return gbyte_trmv( n, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_ger( double m, double n )
    { return m*n; }

inline double fadds_ger( double m, double n )
    { return m*n; }

template< typename T >
inline double gflop_ger( double m, double n, T* x )
{
    return (fmuls_ger(m, n) + fadds_ger(m, n)) / 1e9;
}

template< typename T >
inline double gflop_ger( double m, double n, std::complex<T>* x )
{
    return (6*fmuls_ger(m, n) + 2*fadds_ger(m, n)) / 1e9;
}

template< typename T >
inline double gbyte_ger( double m, double n, T* x )
{
    // read A, x, y; write A
    return (2*m*n + m + n) * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_her( double n, T* x )
{
    return gflop_ger( n, n, x );
}

template< typename T >
inline double gbyte_her( double n, T* x )
{
    // read A triangle, x; write A triangle
    return ((n+1)*n + n) * sizeof(T);
}

template< typename T >
inline double gflop_syr( double n, T* x )
{
    return gflop_her( n, x );
}

template< typename T >
inline double gbyte_syr( double n, T* x )
{
    return gbyte_her( n, x );
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_her2( double n, T* x )
{
    return 2*gflop_ger( n, n, x );
}

template< typename T >
inline double gbyte_her2( double n, T* x )
{
    // read A triangle, x, y; write A triangle
    return ((n+1)*n + n + n) * sizeof(T);
}

template< typename T >
inline double gflop_syr2( double n, T* x )
{
    return gflop_her2( n, x );
}

template< typename T >
inline double gbyte_syr2( double n, T* x )
{
    return gbyte_her2( n, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_gemm( double m, double n, double k )
    { return m*n*k; }

inline double fadds_gemm( double m, double n, double k )
    { return m*n*k; }

template< typename T >
inline double gflop_gemm( double m, double n, double k, T* x )
{
    return (fmuls_gemm(m, n, k) + fadds_gemm(m, n, k)) / 1e9;
}

template< typename T >
inline double gflop_gemm( double m, double n, double k, std::complex<T>* x )
{
    return (6*fmuls_gemm(m, n, k) + 2*fadds_gemm(m, n, k)) / 1e9;
}

template< typename T >
inline double gbyte_gemm( double m, double n, double k, T* x )
{
    // read A, B, C; write C
    return (m*k + k*n + 2*m*n) * sizeof(T);
}


// -----------------------------------------------------------------------------
inline double fmuls_hemm( blas::Side side, double m, double n )
    { return (side == blas::Side::Left ? m*m*n : m*n*n); }

inline double fadds_hemm( blas::Side side, double m, double n )
    { return (side == blas::Side::Left ? m*m*n : m*n*n); }

template< typename T >
inline double gflop_hemm( blas::Side side, double m, double n, T* x )
{
    return (fmuls_hemm(side, m, n) + fadds_hemm(side, m, n)) / 1e9;
}

template< typename T >
inline double gflop_hemm( blas::Side side, double m, double n, std::complex<T>* x )
{
    return (6*fmuls_hemm(side, m, n) + 2*fadds_hemm(side, m, n)) / 1e9;
}

template< typename T >
inline double gbyte_hemm( blas::Side side, double m, double n, T* x )
{
    // read A, B, C; write C
    double sizeA = (side == blas::Side::Left ? 0.5*m*(m+1) : 0.5*n*(n+1));
    return (sizeA + 3*m*n) * sizeof(T);
}

template< typename T >
inline double gflop_symm( blas::Side side, double m, double n, T* x )
{
    return gflop_hemm( side, m, n, x );
}

template< typename T >
inline double gbyte_symm( blas::Side side, double m, double n, T* x )
{
    return gbyte_hemm( side, m, n, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_herk( double n, double k )
    { return 0.5*k*n*(n+1); }

inline double fadds_herk( double n, double k )
    { return 0.5*k*n*(n+1); }

template< typename T >
inline double gflop_herk( double n, double k, T* x )
{
    return (fmuls_herk(n, k) + fadds_herk(n, k)) / 1e9;
}

template< typename T >
inline double gflop_herk( double n, double k, std::complex<T>* x )
{
    return (6*fmuls_herk(n, k) + 2*fadds_herk(n, k)) / 1e9;
}

template< typename T >
inline double gbyte_herk( double n, double k, T* x )
{
    // read A, C; write C
    double sizeC = 0.5*n*(n+1);
    return (n*k + 2*sizeC) * sizeof(T);
}

template< typename T >
inline double gflop_syrk( double n, double k, T* x )
{
    return gflop_herk( n, k, x );
}

template< typename T >
inline double gbyte_syrk( double n, double k, T* x )
{
    return gbyte_herk( n, k, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_her2k( double n, double k )
    { return k*n*n; }

inline double fadds_her2k( double n, double k )
    { return k*n*n; }

template< typename T >
inline double gflop_her2k( double n, double k, T* x )
{
    return (fmuls_her2k(n, k) + fadds_her2k(n, k)) / 1e9;
}

template< typename T >
inline double gflop_her2k( double n, double k, std::complex<T>* x )
{
    return (6*fmuls_her2k(n, k) + 2*fadds_her2k(n, k)) / 1e9;
}

template< typename T >
inline double gbyte_her2k( double n, double k, T* x )
{
    // read A, B, C; write C
    double sizeC = 0.5*n*(n+1);
    return (2*n*k + 2*sizeC) * sizeof(T);
}

template< typename T >
inline double gflop_syr2k( double n, double k, T* x )
{
    return gflop_her2k( n, k, x );
}

template< typename T >
inline double gbyte_syr2k( double n, double k, T* x )
{
    return gbyte_her2k( n, k, x );
}


// -----------------------------------------------------------------------------
inline double fmuls_trmm( blas::Side side, double m, double n )
{
    if (side == blas::Side::Left)
        return 0.5*n*m*(m + 1);
    else
        return 0.5*m*n*(n + 1);
}

inline double fadds_trmm( blas::Side side, double m, double n )
{
    if (side == blas::Side::Left)
        return 0.5*n*m*(m - 1);
    else
        return 0.5*m*n*(n - 1);
}

template< typename T >
inline double gflop_trmm( blas::Side side, double m, double n, T* x )
{
    return (fmuls_trmm(side, m, n) + fadds_trmm(side, m, n)) / 1e9;
}

template< typename T >
inline double gflop_trmm( blas::Side side, double m, double n, std::complex<T>* x )
{
    return (6*fmuls_trmm(side, m, n) + 2*fadds_trmm(side, m, n)) / 1e9;
}

template< typename T >
inline double gbyte_trmm( blas::Side side, double m, double n, T* x )
{
    // read A triangle, x; write x
    if (side == blas::Side::Left)
        return (0.5*(m+1)*m + 2*m*n) * sizeof(T);
    else
        return (0.5*(n+1)*n + 2*m*n) * sizeof(T);
}


// -----------------------------------------------------------------------------
template< typename T >
inline double gflop_trsm( blas::Side side, double m, double n, T* x )
{
    return gflop_trmm( side, m, n, x );
}

template< typename T >
inline double gbyte_trsm( blas::Side side, double m, double n, T* x )
{
    return gbyte_trmm( side, m, n, x );
}

//==============================================================================
// template class. Example:
// gflop< float >::gemm( m, n, k ) yields flops for sgemm.
// gflop< std::complex<float> >::gemm( m, n, k ) yields flops for cgemm.
//==============================================================================
template< typename T >
class Gbyte
{
public:
    // ----------------------------------------
    // Level 1 BLAS
    static double asum( double n )
    {
        // read x
        return 1e-9 * (n * sizeof(T));
    }

    static double axpy( double n )
    {
        // read x, y; write y
        return 1e-9 * (3*n * sizeof(T));
    }

    static double copy( double n )
    {
        // read x; write y
        return 1e-9 * (2*n * sizeof(T));
    }

    static double iamax( double n )
    {
        // read x
        return 1e-9 * (n * sizeof(T));
    }

    static double nrm2( double n )
    {
        // read x
        return 1e-9 * (n * sizeof(T));
    }

    static double dot( double n )
    {
        // read x, y
        return 1e-9 * (2*n * sizeof(T));
    }

    static double scal( double n )
    {
        // read x; write x
        return 1e-9 * (2*n * sizeof(T));
    }

    static double swap( double n )
    {
        // read x, y; write x, y
        return 1e-9 * (4*n * sizeof(T));
    }

    // ----------------------------------------
    // Level 2 BLAS
    static double gemv( double m, double n )
    {
        // read A, x; write y
        return 1e-9 * ((m*n + m + n) * sizeof(T));
    }

    static double hemv( double n )
    {
        // read A triangle, x; write y
        return 1e-9 * ((0.5*(n+1)*n + 2*n) * sizeof(T));
    }

    static double symv( double n )
    {
        return hemv( n );
    }

    static double trmv( double n )
    {
        // read A triangle, x; write x
        return 1e-9 * ((0.5*(n+1)*n + 2*n) * sizeof(T));
    }

    static double trsv( double n )
    {
        return trmv( n );
    }

    static double ger( double m, double n )
    {
        // read A, x, y; write A
        return 1e-9 * ((2*m*n + m + n) * sizeof(T));
    }

    static double her( double n )
    {
        // read A triangle, x; write A triangle
        return 1e-9 * (((n+1)*n + n) * sizeof(T));
    }

    static double syr( double n )
    {
        return her( n );
    }

    static double her2( double n )
    {
        // read A triangle, x, y; write A triangle
        return 1e-9 * (((n+1)*n + n + n) * sizeof(T));
    }

    static double syr2( double n )
    {
        return her2( n );
    }

    // ----------------------------------------
    // Level 3 BLAS
    static double gemm( double m, double n, double k )
    {
        // read A, B, C; write C
        return 1e-9 * ((m*k + k*n + 2*m*n) * sizeof(T));
    }

    static double hemm( blas::Side side, double m, double n )
    {
        // read A, B, C; write C
        double sizeA = (side == blas::Side::Left ? 0.5*m*(m+1) : 0.5*n*(n+1));
        return 1e-9 * ((sizeA + 3*m*n) * sizeof(T));
    }

    static double symm( blas::Side side, double m, double n )
    {
        return hemm( side, m, n );
    }

    static double herk( double n, double k )
    {
        // read A, C; write C
        double sizeC = 0.5*n*(n+1);
        return 1e-9 * ((n*k + 2*sizeC) * sizeof(T));
    }

    static double syrk( double n, double k )
    {
        return herk( n, k );
    }

    static double her2k( double n, double k )
    {
        // read A, B, C; write C
        double sizeC = 0.5*n*(n+1);
        return 1e-9 * ((2*n*k + 2*sizeC) * sizeof(T));
    }

    static double syr2k( double n, double k )
    {
        return her2k( n, k );
    }

    static double trmm( blas::Side side, double m, double n )
    {
        // read A triangle, x; write x
        if (side == blas::Side::Left)
            return 1e-9 * ((0.5*(m+1)*m + 2*m*n) * sizeof(T));
        else
            return 1e-9 * ((0.5*(n+1)*n + 2*m*n) * sizeof(T));
    }

    static double trsm( blas::Side side, double m, double n )
    {
        return trmm( side, m, n );
    }
};

//==============================================================================
// template class. Example:
// gflop< float >::gemm( m, n, k ) yields flops for sgemm.
// gflop< std::complex<float> >::gemm( m, n, k ) yields flops for cgemm.
//==============================================================================
template< typename T >
class Gflop
{
public:
    // ----------------------------------------
    // Level 1 BLAS
    static double asum( double n, std::complex<T>* x )
    {
        return (6*fmuls_asum(n) + 2*fadds_asum(n)) / 1e9;
    }

    // ----------------------------------------
    // Level 2 BLAS
    static double gemv(double m, double n)
        { return 1e-9 * (fmuls_gemv(m, n) + fadds_gemv(m, n)); }

    static double symv(double n)
        { return gemv( n, n ); }

    static double hemv(double n)
        { return symv( n, n ); }

    // ----------------------------------------
    // Level 3 BLAS
    static double gemm(double m, double n, double k)
        { return 1e-9 * (fmuls_gemm(m, n, k) + fadds_gemm(m, n, k)); }

    static double hemm(blas::Side side, double m, double n)
        { return 1e-9 * (fmuls_hemm(side, m, n) + fadds_hemm(side, m, n)); }

    static double symm(blas::Side side, double m, double n)
        { return hemm( side, m, n ); }

    static double herk(double n, double k)
        { return 1e-9 * (fmuls_herk(n, k) + fadds_herk(n, k)); }

    static double syrk(double n, double k)
        { return syrk( n, k ); }

    static double her2k(double n, double k)
        { return 1e-9 * (fmuls_her2k(n, k) + fadds_her2k(n, k)); }

    static double syr2k(double n, double k)
        { return syr2k( n, k ); }

    static double trmm(blas::Side side, double m, double n)
        { return 1e-9 * (fmuls_trmm(side, m, n) + fadds_trmm(side, m, n)); }

    static double trsm(blas::Side side, double m, double n)
        { return trmm( side, m, n ); }
};

//==============================================================================
// specialization for complex
// flops = 6*muls + 2*adds
//==============================================================================
template< typename T >
class Gflop< std::complex<T> >
{
public:
    // ----------------------------------------
    // Level 1 BLAS

    // ----------------------------------------
    // Level 2 BLAS
    static double gemv(double m, double n)
        { return 1e-9 * (6*fmuls_gemv(m, n) + 2*fadds_gemv(m, n)); }

    static double hemv(double n)
        { return gemv( n, n ); }

    static double symv(double n)
        { return hemv( n ); }

    // ----------------------------------------
    // Level 3 BLAS
    static double gemm(double m, double n, double k)
        { return 1e-9 * (6*fmuls_gemm(m, n, k) + 2*fadds_gemm(m, n, k)); }

    static double hemm(blas::Side side, double m, double n)
        { return 1e-9 * (6*fmuls_hemm(side, m, n) + 2*fadds_hemm(side, m, n)); }

    static double symm(blas::Side side, double m, double n)
        { return hemm( side, m, n ); }

    static double herk(double n, double k)
        { return 1e-9 * (6*fmuls_herk(n, k) + 2*fadds_herk(n, k)); }

    static double syrk(double n, double k)
        { return herk( n, k ); }

    static double her2k(double n, double k)
        { return 1e-9 * (6*fmuls_her2k(n, k) + 2*fadds_her2k(n, k)); }

    static double syr2k(double n, double k)
        { return her2k( n, k ); }

    static double trmm(blas::Side side, double m, double n)
        { return 1e-9 * (6*fmuls_trmm(side, m, n) + 2*fadds_trmm(side, m, n)); }

    static double trsm(blas::Side side, double m, double n)
        { return trmm( side, m, n ); }
};

}  // namespace blas

#endif        //  #ifndef FLOPS_HH
