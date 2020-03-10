#ifndef BLAS_HH
#define BLAS_HH

#include "blas/wrappers.hh"

// =============================================================================
// Level 1 BLAS template implementations

#include "blas/asum.hh"
#include "blas/axpy.hh"
#include "blas/copy.hh"
#include "blas/dot.hh"
#include "blas/iamax.hh"
#include "blas/nrm2.hh"
#include "blas/rot.hh"
#include "blas/rotg.hh"
#include "blas/rotm.hh"
#include "blas/rotmg.hh"
#include "blas/scal.hh"
#include "blas/swap.hh"

// =============================================================================
// Level 2 BLAS template implementations

#include "blas/gemv.hh"
#include "blas/ger.hh"
#include "blas/geru.hh"
#include "blas/hemv.hh"
#include "blas/her.hh"
#include "blas/her2.hh"
#include "blas/symv.hh"
#include "blas/syr.hh"
#include "blas/syr2.hh"
#include "blas/trmv.hh"
#include "blas/trsv.hh"

// =============================================================================
// Level 3 BLAS template implementations

#include "blas/gemm.hh"
#include "blas/hemm.hh"
#include "blas/herk.hh"
#include "blas/her2k.hh"
#include "blas/symm.hh"
#include "blas/syrk.hh"
#include "blas/syr2k.hh"
#include "blas/trmm.hh"
#include "blas/trsm.hh"

// =============================================================================
// Device BLAS

#ifdef BLASPP_WITH_CUBLAS
#include "blas/device_blas.hh"
#endif

#endif        //  #ifndef BLAS_HH
