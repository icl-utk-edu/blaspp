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
// Convert enum to LAPACK-style string.
inline const char* layout2str( Layout layout )
{
    switch (layout) {
        case Layout::ColMajor: return "col";
        case Layout::RowMajor: return "row";
    }
    return "";
}

inline const char* op2str( Op op )
{
    switch (op) {
        case Op::NoTrans:   return "notrans";
        case Op::Trans:     return "trans";
        case Op::ConjTrans: return "conj";
    }
    return "";
}

inline const char* uplo2str( Uplo uplo )
{
    switch (uplo) {
        case Uplo::Lower: return "lower";
        case Uplo::Upper: return "upper";
    }
    return "";
}

inline const char* diag2str( Diag diag )
{
    switch (diag) {
        case Diag::NonUnit: return "nonunit";
        case Diag::Unit:    return "unit";
    }
    return "";
}

inline const char* side2str( Side side )
{
    switch (side) {
        case Side::Left:  return "left";
        case Side::Right: return "right";
    }
    return "";
}

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
/// Exception class for BLAS errors.
class Error: public std::exception {
public:
    /// Constructs BLAS error
    Error():
        std::exception()
    {}

    /// Constructs BLAS error with message: "func: msg"
    Error( const char* msg, const char* func ):
        std::exception(),
        msg_( std::string(func) + ": " + msg )
    {}

    /// Returns BLAS error message
    virtual const char* what() { return msg_.c_str(); }

private:
    std::string msg_;
};

// -----------------------------------------------------------------------------
// Extend real, imag, conj to other datatypes.
template< typename T >
inline T real( T x ) { return x; }

template< typename T >
inline T imag( T x ) { return 0; }

template< typename T >
inline T conj( T x ) { return x; }

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
//  traits
/// Given a type, defines corresponding real and complex types.
/// E.g., for float,          real_t = float, complex_t = std::complex<float>,
///       for complex<float>, real_t = float, complex_t = std::complex<float>.

template< typename T >
class traits
{
public:
    typedef T real_t;
    typedef std::complex<T> complex_t;
};

// ----------------------------------------
template< typename T >
class traits< std::complex<T> >
{
public:
    typedef T real_t;
    typedef std::complex<T> complex_t;
};

// -----------------------------------------------------------------------------
//  traits2
/// Given two types, defines scalar and real types compatible with both types.
/// E.g., for pair (float, complex<float>),
/// scalar_t = complex<float>, real_t = float.

// By default, scalars and reals are T1.
// Later classes specialize if it should be T2 or something else
template< typename T1, typename T2 >
class traits2
{
public:
    typedef T1 scalar_t;
    typedef T1 real_t;
};

// ----------------------------------------
// int
template<>
class traits2< int, int64_t >
{
public:
    typedef int64_t scalar_t;
    typedef int64_t real_t;
};

// ---------------
template<>
class traits2< int, float >
{
public:
    typedef float scalar_t;
    typedef float real_t;
};

// ---------------
template<>
class traits2< int, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// float
template<>
class traits2< float, double >
{
public:
    typedef double scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< float, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

// ---------------
template<>
class traits2< float, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// double
template<>
class traits2< double, std::complex<float> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ---------------
template<>
class traits2< double, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// ----------------------------------------
// complex<float>
template<>
class traits2< std::complex<float>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

template<>
class traits2< std::complex<float>, std::complex<float> >
{
public:
    typedef std::complex<float> scalar_t;
    typedef float real_t;
};

// ----------------------------------------
// complex<double>
template<>
class traits2< std::complex<double>, std::complex<double> >
{
public:
    typedef std::complex<double> scalar_t;
    typedef double real_t;
};

// -----------------------------------------------------------------------------
// traits3
/// Given three types, defines scalar and real types compatible with all types.
/// E.g., for the triple (float, complex<float>, double),
/// scalar_t = complex<double>, real_t = double.

// ----------------------------------------
template< typename T1, typename T2, typename T3 >
class traits3
{
public:
    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::scalar_t scalar_t;

    typedef typename
        traits2< typename traits2<T1,T2>::scalar_t, T3 >::real_t real_t;
};

// -----------------------------------------------------------------------------
// max that works with different data types, e.g., max( int, int64_t )
template< typename T1, typename T2 >
typename blas::traits2< T1, T2 >::scalar_t
max( T1 a, T2 b )
{
    return (a >= b ? a : b);
}

// -----------------------------------------------------------------------------
// min that works with different data types, e.g., min( int, int64_t )
template< typename T1, typename T2 >
typename blas::traits2< T1, T2 >::scalar_t
min( T1 a, T2 b )
{
    return (a <= b ? a : b);
}

namespace internal {

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// called by throw_if_ macro
inline void throw_if( bool cond, const char* condstr, const char* func )
{
    if (cond) {
        throw Error( condstr, func );
    }
}

// -----------------------------------------------------------------------------
// internal helper function; throws Error if cond is true
// uses printf-style format for error message
// called by throw_if_msg_ macro
inline void throw_if( bool cond, const char* condstr, const char* func, const char* format, ... )
{
    if (cond) {
        char buf[80];
        va_list va;
        va_start( va, format );
        vsnprintf( buf, sizeof(buf), format, va );
        throw Error( buf, func );
    }
}

} // namespace internal

// internal macro to get string #cond; throws Error if cond is true
// ex: throw_if_( a < b );
#define throw_if_( cond ) \
    internal::throw_if( cond, #cond, __func__ )

// internal macro takes cond and printf-style format for error message.
// throws Error if cond is true.
// ex: throw_if_msg_( a < b, "a %d < b %d", a, b );
#define throw_if_msg_( cond, ... ) \
    internal::throw_if( cond, #cond, __func__, __VA_ARGS__ )

}  // namespace blas

#endif        //  #ifndef BLAS_UTIL_HH
