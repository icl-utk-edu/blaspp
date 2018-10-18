#ifndef TEST_HH
#define TEST_HH

#include "libtest.hh"
#include "blas.hh"

// -----------------------------------------------------------------------------
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
    //libtest::ParamDouble tol;  // stricter bounds don't need arbitrary tol
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

// -----------------------------------------------------------------------------
// Level 1 BLAS
void test_asum  ( Params& params, bool run );
void test_axpy  ( Params& params, bool run );
void test_copy  ( Params& params, bool run );
void test_dot   ( Params& params, bool run );
void test_dotu  ( Params& params, bool run );
void test_iamax ( Params& params, bool run );
void test_nrm2  ( Params& params, bool run );
void test_rot   ( Params& params, bool run );
void test_rotg  ( Params& params, bool run );
void test_rotm  ( Params& params, bool run );
void test_rotmg ( Params& params, bool run );
void test_scal  ( Params& params, bool run );
void test_swap  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 2 BLAS
void test_gemv  ( Params& params, bool run );
void test_ger   ( Params& params, bool run );
void test_geru  ( Params& params, bool run );
void test_hemv  ( Params& params, bool run );
void test_her   ( Params& params, bool run );
void test_her2  ( Params& params, bool run );
void test_symv  ( Params& params, bool run );
void test_syr   ( Params& params, bool run );
void test_syr2  ( Params& params, bool run );
void test_trmv  ( Params& params, bool run );
void test_trsv  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 BLAS
void test_gemm  ( Params& params, bool run );
void test_hemm  ( Params& params, bool run );
void test_her2k ( Params& params, bool run );
void test_herk  ( Params& params, bool run );
void test_symm  ( Params& params, bool run );
void test_syr2k ( Params& params, bool run );
void test_syrk  ( Params& params, bool run );
void test_trmm  ( Params& params, bool run );
void test_trsm  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 Batch BLAS
void test_batch_gemm  ( Params& params, bool run );
void test_batch_hemm  ( Params& params, bool run );
void test_batch_her2k ( Params& params, bool run );
void test_batch_herk  ( Params& params, bool run );
void test_batch_symm  ( Params& params, bool run );
void test_batch_syr2k ( Params& params, bool run );
void test_batch_syrk  ( Params& params, bool run );
void test_batch_trmm  ( Params& params, bool run );
void test_batch_trsm  ( Params& params, bool run );

// -----------------------------------------------------------------------------
// Level 3 GPU BLAS
#ifdef BLASPP_WITH_CUBLAS
void test_gemm_device  ( Params& params, bool run );
void test_trsm_device  ( Params& params, bool run );
void test_trmm_device  ( Params& params, bool run );
void test_hemm_device  ( Params& params, bool run );
void test_symm_device  ( Params& params, bool run );
void test_herk_device  ( Params& params, bool run );
void test_syrk_device  ( Params& params, bool run );
void test_her2k_device  ( Params& params, bool run );
void test_syr2k_device  ( Params& params, bool run );

void test_batch_gemm_device( Params& params, bool run );
void test_batch_trsm_device( Params& params, bool run );
void test_batch_trmm_device( Params& params, bool run );
void test_batch_hemm_device( Params& params, bool run );
void test_batch_symm_device( Params& params, bool run );
void test_batch_herk_device( Params& params, bool run );
void test_batch_syrk_device( Params& params, bool run );
void test_batch_her2k_device( Params& params, bool run );
void test_batch_syr2k_device( Params& params, bool run );
#endif

// -----------------------------------------------------------------------------
// auxiliary
void test_error ( Params& params, bool run );
void test_max   ( Params& params, bool run );

#endif  //  #ifndef TEST_HH
