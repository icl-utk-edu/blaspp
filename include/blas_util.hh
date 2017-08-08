#ifndef BLAS_UTIL_HH
#define BLAS_UTIL_HH

#include <exception>
#include <complex>

#include <assert.h>

namespace blas {

// -----------------------------------------------------------------------------
enum class Layout : char { ColMajor = 'C', RowMajor = 'R' };
enum class Op     : char { NoTrans  = 'N', Trans    = 'T', ConjTrans = 'C' };
enum class Uplo   : char { Upper    = 'U', Lower    = 'L' };
enum class Diag   : char { NonUnit  = 'N', Unit     = 'U' };
enum class Side   : char { Left     = 'L', Right    = 'R' };

// -----------------------------------------------------------------------------
// Convert enum to LAPACK-style char.
inline char layout2char( Layout layout ) { return char(layout); }
inline char     op2char( Op     op     ) { return char(op);     }
inline char   uplo2char( Uplo   uplo   ) { return char(uplo);   }
inline char   diag2char( Diag   diag   ) { return char(diag);   }
inline char   side2char( Side   side   ) { return char(side);   }

// -----------------------------------------------------------------------------
// Convert LAPACK-style char to enum.
inline Layout char2layout( char layout )
{
    layout = (char) toupper( layout );
    assert( layout == 'C' || layout == 'R' );
    return Layout( layout );
}

inline Op char2op( char op )
{
    op = (char) toupper( op );
    assert( op == 'N' || op == 'T' || op == 'C' );
    return Op( op );
}

inline Uplo char2uplo( char uplo )
{
    uplo = (char) toupper( uplo );
    assert( uplo == 'L' || uplo == 'U' );
    return Uplo( uplo );
}

inline Diag char2diag( char diag )
{
    diag = (char) toupper( diag );
    assert( diag == 'N' || diag == 'U' );
    return Diag( diag );
}

inline Side char2side( char side )
{
    side = (char) toupper( side );
    assert( side == 'L' || side == 'R' );
    return Side( side );
}


// -----------------------------------------------------------------------------
class Error: public std::exception {
public:
    Error(): std::exception() {}
    Error( const char* msg ): std::exception(), msg_( msg ) {}
    virtual const char* what() { return msg_.c_str(); }
private:
    std::string msg_;
};


// -----------------------------------------------------------------------------
// Extend real, imag, conj to other datatypes.
inline int    real( int    x ) { return x; }
inline float  real( float  x ) { return x; }
inline double real( double x ) { return x; }

inline int    imag( int    x ) { return 0; }
inline float  imag( float  x ) { return 0; }
inline double imag( double x ) { return 0; }

inline int    conj( int    x ) { return x; }
inline float  conj( float  x ) { return x; }
inline double conj( double x ) { return x; }


// -----------------------------------------------------------------------------
// 1-norm absolute value, |Re(x)| + |Im(x)|
template< typename T >
T abs1( T x )
{
    return std::abs( x );
}

template< typename T >
T abs1( std::complex<T> x )
{
    return std::abs( real(x) ) + std::abs( imag(x) );
}

// -----------------------------------------------------------------------------
// traits: given a type, defines its norm type

// ----------------------------------------
template< typename T >
class traits
{
public:
    typedef T norm_t;
};

// ----------------------------------------
template< typename T >
class traits< std::complex<T> >
{
public:
    typedef T norm_t;
};


// -----------------------------------------------------------------------------
// traits2: given 2 types, defines their scalar and norm types.
// Default is type T1, then overrides are given for cases where it should be T2
// or something different.

// ----------------------------------------
template< typename T1, typename T2 >
class traits2
{
public:
    typedef T1 scalar_t;
    typedef T1 norm_t;
};

// ----------------------------------------
// float
template<>
class traits2< float, double >
{
public:
    typedef double scalar_t;
    typedef double norm_t;
};

// ---------------
template<>
class traits2< float, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float norm_t;
};

// ---------------
template<>
class traits2< float, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double norm_t;
};

// ----------------------------------------
// double
template<>
class traits2< double, std::complex<float> >
{
public:
    // TODO: what should this be? do we care?
    typedef std::complex<double> scalar_t;
    typedef double norm_t;
};

// ---------------
template<>
class traits2< double, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double norm_t;
};

// ----------------------------------------
// complex<float>
template<>
class traits2< std::complex<float>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double norm_t;
};


// -----------------------------------------------------------------------------
// traits2: given 3 types, defines their scalar and norm types.

// ----------------------------------------
template< typename T1, typename T2, typename T3 >
class traits3
{
public:
    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::scalar_t scalar_t;

    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::norm_t norm_t;
};


// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by throw_if_ macro
inline void throw_if__( bool cond, const char* condstr )
{
    if (cond) {
        throw Error( condstr );
    }
}

// internal macro to get string #cond; throws Error if cond is true
#define throw_if_( cond ) \
    throw_if__( cond, #cond )

}  // namespace blas

#endif        //  #ifndef BLAS_UTIL_HH
