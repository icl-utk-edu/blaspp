#ifndef TEST_DEVICE_HH
#define TEST_DEVICE_HH

#include "libtest.hh"
#include "blas_util.hh"
#include "test_common.hh"

// -----------------------------------------------------------------------------
// Level 1 BLAS

// -----------------------------------------------------------------------------
// Level 2 BLAS

// -----------------------------------------------------------------------------
// Level 3 BLAS
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

// -----------------------------------------------------------------------------

#endif  //  #ifndef TEST_DEVIC_HH
