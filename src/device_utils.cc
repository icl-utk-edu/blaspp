#include "blas/device.hh"

// -----------------------------------------------------------------------------
// set device
void blas::set_device(blas::Device device)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaSetDevice((device_blas_int)device) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif
}

// -----------------------------------------------------------------------------
// get current device
void blas::get_device(blas::Device *device)
{
    device_blas_int dev;

    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaGetDevice(&dev) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif

    (*device) = (blas::Device)dev;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device trans constant
device_trans_t    blas::device_trans_const(blas::Op trans)
{
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );

    device_trans_t trans_ = DevNoTrans;
    switch (trans) {
        case Op::NoTrans:   trans_ = DevNoTrans;   break;
        case Op::Trans:     trans_ = DevTrans;     break;
        case Op::ConjTrans: trans_ = DevConjTrans; break;
        default:;
    }
    return trans_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device diag constant
device_diag_t    blas::device_diag_const(blas::Diag diag)
{
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );

    device_diag_t diag_ = DevDiagUnit;
    switch (diag) {
        case Diag::Unit:    diag_ = DevDiagUnit;    break;
        case Diag::NonUnit: diag_ = DevDiagNonUnit; break;
        default:;
    }
    return diag_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device uplo constant
device_uplo_t    blas::device_uplo_const(blas::Uplo uplo)
{
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );

    device_uplo_t uplo_ = DevUploLower;
    switch (uplo) {
        case Uplo::Upper: uplo_ = DevUploUpper; break;
        case Uplo::Lower: uplo_ = DevUploLower; break;
        default:;
    }
    return uplo_;
}

// -----------------------------------------------------------------------------
/// @return the corresponding device side constant
device_side_t    blas::device_side_const(blas::Side side)
{
    blas_error_if( side != Side::Left &&
                   side != Side::Right );

    device_side_t side_ = DevSideLeft;
    switch (side) {
        case Side::Left:  side_ = DevSideLeft;  break;
        case Side::Right: side_ = DevSideRight; break;
        default:;
    }
    return side_;
}

// -----------------------------------------------------------------------------
/// @free a device pointer
void blas::device_free(void* ptr)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaFree( ptr ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: free memory for AMD GPUs
    #endif
}

// -----------------------------------------------------------------------------
/// @free a pinned memory space
void blas::device_free_pinned(void* ptr)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaFreeHost( ptr ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: free memory using AMD driver API
    #endif
}
