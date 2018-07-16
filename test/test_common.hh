#ifndef TEST_COMMON_HH
#define TEST_COMMON_HH

#include "blas.hh"
#include "libtest.hh"

class Params: public libtest::ParamsBase
{
public:
    const double inf = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double pi  = 3.141592653589793;
    const double e   = 2.718281828459045;

    Params();

    // Field members are explicitly public.
    // Order here determines output order.
    // ----- test framework parameters
    libtest::ParamChar   check;
    libtest::ParamChar   ref;
    libtest::ParamDouble tol;
    libtest::ParamInt    repeat;
    libtest::ParamInt    verbose;
    libtest::ParamInt    cache;

    // ----- routine parameters
    libtest::ParamEnum< libtest::DataType > datatype;
    libtest::ParamEnum< blas::Layout >      layout;
    libtest::ParamEnum< blas::Side >        side;
    libtest::ParamEnum< blas::Uplo >        uplo;
    libtest::ParamEnum< blas::Op >          trans;
    libtest::ParamEnum< blas::Op >          transA;
    libtest::ParamEnum< blas::Op >          transB;
    libtest::ParamEnum< blas::Diag >        diag;

    libtest::ParamInt3   dim;
    libtest::ParamDouble alpha;
    libtest::ParamDouble beta;
    libtest::ParamInt    incx;
    libtest::ParamInt    incy;
    libtest::ParamInt    align;
    libtest::ParamInt    batch;
    libtest::ParamInt    device;

    // ----- output parameters
    libtest::ParamScientific error;
    libtest::ParamDouble     time;
    libtest::ParamDouble     gflops;
    libtest::ParamDouble     gbytes;

    libtest::ParamDouble     ref_time;
    libtest::ParamDouble     ref_gflops;
    libtest::ParamDouble     ref_gbytes;

    libtest::ParamOkay       okay;
};

// -----------------------------------------------------------------------------
template< typename T >
inline T roundup( T x, T y )
{
    return T( (x + y - 1) / y ) * y;
}

// -----------------------------------------------------------------------------
#define assert_throw( expr, exception_type ) \
    try { \
        expr; \
        fprintf( stderr, "Error: didn't throw expected exception at %s:%d\n", \
                 __FILE__, __LINE__ ); \
        throw std::exception(); \
    } \
    catch (exception_type& err) { \
        if (verbose >= 3) { \
            printf( "Caught expected exception: %s\n", err.what() ); \
        } \
    }

#endif    // TEST_COMMON_HH
