#ifndef TEST_HH
#define TEST_HH

#include "libtest.hh"
#include "blas_util.hh"
#include "test_common.hh"

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
// auxiliary
void test_error ( Params& params, bool run );
void test_max   ( Params& params, bool run );

#endif  //  #ifndef TEST_HH
