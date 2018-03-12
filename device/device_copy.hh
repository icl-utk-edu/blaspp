#ifndef DEVICE_COPY_HH
#define DEVICE_COPY_HH

namespace blas {

template<typename T> 
inline void device_setmatrix(int64_t m, int64_t n, T* hostPtr, int64_t ldh, T* devPtr, int64_t ldd, Queue &queue);

template<typename T>
inline void device_getmatrix(int64_t m, int64_t n, T* devPtr, int64_t ldd, T* hostPtr, int64_t ldh, Queue &queue);

template<typename T> 
inline void device_setvector(int64_t n, T* hostPtr, int64_t inch, T* devPtr, int64_t incd, Queue &queue);

template<typename T> 
inline void device_getvector(int64_t n, T* devPtr, int64_t incd, T* hostPtr, int64_t inch, Queue &queue);

}        //  namespace blas

template<typename T> 
inline 
void blas::device_setmatrix(int64_t m, int64_t n, T* hostPtr, int64_t ldh, T* devPtr, int64_t ldd, Queue &queue){
    #ifdef HAVE_CUBLAS
    device_blas_check( cublasSetMatrixAsync( 
                       (device_blas_int)m,    (device_blas_int)n, (device_blas_int)sizeof(T), 
                       (const void *)hostPtr, (device_blas_int)ldh, 
                       (      void *)devPtr,  (device_blas_int)ldd, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_set_matrix
    #endif
}

template<typename T>
inline 
void blas::device_getmatrix(int64_t m, int64_t n, T* devPtr, int64_t ldd, T* hostPtr, int64_t ldh, Queue &queue){
    #ifdef HAVE_CUBLAS
    device_blas_check( cublasGetMatrixAsync( 
                       (device_blas_int)m,    (device_blas_int)n, (device_blas_int)sizeof(T), 
                       (const void *)devPtr,  (device_blas_int)ldd, 
                       (      void *)hostPtr, (device_blas_int)ldh, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_get_matrix
    #endif
}

template<typename T> 
inline 
void blas::device_setvector(int64_t n, T* hostPtr, int64_t inch, T* devPtr, int64_t incd, Queue &queue){
    #ifdef HAVE_CUBLAS
    device_blas_check( cublasSetVectorAsync( 
                       (device_blas_int)n,    (device_blas_int)sizeof(T), 
                       (const void *)hostPtr, (device_blas_int)inch, 
                       (      void *)devPtr,  (device_blas_int)incd, queue.stream() ) );
    #elif defined(ROCBLAS)
    // TODO: call rocblas_set_vector
    #endif
}

template<typename T> 
inline 
void blas::device_getvector(int64_t n, T* devPtr, int64_t incd, T* hostPtr, int64_t inch, Queue &queue){
    #ifdef HAVE_CUBLAS
    device_blas_check( cublasGetVectorAsync(
                       (device_blas_int)n,    (device_blas_int)sizeof(T), 
                       (const void *)devPtr,  (device_blas_int)incd, 
                       (      void *)hostPtr, (device_blas_int)inch, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_get_vector
    #endif
}

#endif        //  #ifndef DEVICE_COPY_HH

