#ifndef DEVICE_UTILS_HH
#define DEVICE_UTILS_HH

namespace blas {

typedef int64_t Device; 

inline void set_device(blas::Device device);

inline void get_device(blas::Device *device);

// TODO: errors/exceptions

// functions that converts blaspp constants to device-specific constants
inline device_trans_t   device_trans_const(blas::Op trans);

inline device_diag_t    device_diag_const(blas::Diag diag);

inline device_uplo_t    device_uplo_const(blas::Uplo uplo);

inline device_side_t    device_side_const(blas::Side side);

template<typename T> 
inline  T* device_malloc(int64_t nelements);

inline void device_free(void* ptr);

}        //  namespace blas


inline 
void blas::set_device(blas::Device device){
    #ifdef HAVE_CUBLAS
    device_error_check( cudaSetDevice((device_blas_int)device) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif
}

inline 
void blas::get_device(blas::Device *device){
    device_blas_int dev; 

    #ifdef HAVE_CUBLAS
    device_error_check( cudaGetDevice(&dev) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif

    (*device) = (blas::Device)dev;
}

/// @return the corresponding device trans constant 
inline 
device_trans_t    blas::device_trans_const(blas::Op trans){    
    blas_error_if( trans != Op::NoTrans &&
                   trans != Op::Trans &&
                   trans != Op::ConjTrans );

    device_trans_t trans_ = DevNoTrans; 
    switch(trans)
    {
        case Op::NoTrans:   trans_ = DevNoTrans;   break;
        case Op::Trans:     trans_ = DevTrans;     break;
        case Op::ConjTrans: trans_ = DevConjTrans; break;
        default:;
    }
    return trans_;
}

/// @return the corresponding device diag constant 
inline 
device_diag_t    blas::device_diag_const(blas::Diag diag){
    blas_error_if( diag != Diag::NonUnit &&
                   diag != Diag::Unit );

    device_diag_t diag_ = DevDiagUnit;
    switch(diag)
    {
        case Diag::Unit:    diag_ = DevDiagUnit;    break;
        case Diag::NonUnit: diag_ = DevDiagNonUnit; break;
        default:;
    }
    return diag_;
}

/// @return the corresponding device uplo constant 
inline 
device_uplo_t    blas::device_uplo_const(blas::Uplo uplo){
    blas_error_if( uplo != Uplo::Lower &&
                   uplo != Uplo::Upper );

    device_uplo_t uplo_ = DevUploLower;
    switch(uplo)
    {
        case Uplo::Upper: uplo_ = DevUploUpper; break;
        case Uplo::Lower: uplo_ = DevUploLower; break;
        default:;
    }
    return uplo_;
}

/// @return the corresponding device side constant 
inline 
device_side_t    blas::device_side_const(blas::Side side){
    blas_error_if( side != Side::Left &&
                   side != Side::Right );

    device_side_t side_ = DevSideLeft;
    switch(side)
    {
        case Side::Left:  side_ = DevSideLeft;  break;
        case Side::Right: side_ = DevSideRight; break;
        default:;
    }
    return side_;
}

/// @return a device pointer to an allocated memory space 
template<typename T>
inline 
T* blas::device_malloc(int64_t nelements){
    T* ptr = NULL;
    #ifdef HAVE_CUBLAS
    device_error_check( cudaMalloc((void**)&ptr, nelements * sizeof(T)) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: allocation for AMD GPUs
    #endif
    return ptr;
}

/// @free a device pointer 
inline 
void blas::device_free(void* ptr){
    #ifdef HAVE_CUBLAS
    device_error_check( cudaFree( ptr ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: free memory for AMD GPUs
    #endif
}

#endif        //  #ifndef DEVICE_UTILS_HH

