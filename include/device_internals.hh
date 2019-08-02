#ifndef DEVICE_INTERNALS_HH
#define DEVICE_INTERNALS_HH

namespace blas {

typedef int64_t Device;
#define DEV_QUEUE_DEFAULT_BATCH_LIMIT  (50000)
#define DEV_QUEUE_FORK_SIZE            (10)

// -----------------------------------------------------------------------------
// device queue
class Queue
{
public:
     Queue();
     Queue(blas::Device device, int64_t batch_size);
    ~Queue();

    blas::Device           device();
    device_blas_handle_t   handle();
    void                   sync();
    size_t                 get_batch_limit();
    void**                 get_devPtrArray();

    void     fork();    // switch from default stream to parallel streams
    void     join();    // switch back to the default stream
    void     revolve(); // return the next-in-line stream (for both default and fork modes)

    #ifdef BLASPP_WITH_CUBLAS
    cudaStream_t     stream();
    #elif defined(HAVE_ROCBLAS)
    // TODO: add similar functionality for rocBLAS, if required
    #endif

private:
    blas::Device          device_;               // associated device ID
    device_blas_handle_t  handle_;               // associated device blas handle
    size_t                batch_limit_;          // max workspace allocated for a batch argument in a single call (e.g. a pointer array)
    void**                devPtrArray;           // workspace for pointer arrays of batch routines

    #ifdef BLASPP_WITH_CUBLAS
    size_t           num_active_streams_;                       // the number of streams the queue is currently using for launching kernels (1 by default)
    size_t           current_stream_index_;                     // an index to the current stream in use
    cudaStream_t     *current_stream_;                          // pointer to current stream (default or fork mode)
    cudaStream_t     default_stream_;                           // default CUDA stream for this queue; may be NULL
    cudaStream_t     parallel_streams_[DEV_QUEUE_FORK_SIZE];    // parallel streams in fork mode

    cudaEvent_t      default_event_;
    cudaEvent_t      parallel_events_[DEV_QUEUE_FORK_SIZE];
    #elif defined(HAVE_ROCBLAS)
    // TODO: stream for rocBLAS
    #endif

};

// -----------------------------------------------------------------------------
// device errors
bool is_device_error(device_error_t error);

bool is_device_error(device_blas_status_t status);

const char* device_error_string(device_error_t error);

const char* device_error_string(device_blas_status_t status);

#if defined(BLAS_ERROR_NDEBUG) || (defined(BLAS_ERROR_ASSERT) && defined(NDEBUG))

    // blaspp does no error checking on device errors;
    #define device_error_check( error ) \
        ((void)0)

    // blaspp does no status checking on device blas errors;
    #define device_blas_check( status ) \
        ((void)0)

#elif defined(BLAS_ERROR_ASSERT)

    // blaspp aborts on device errors
    #define device_error_check( error ) \
        do { \
            device_error_t e = error; \
            blas::internal::abort_if( blas::is_device_error(e), __func__, \
                                      "%s", blas::device_error_string(e) ); \
        } while(0)

    // blaspp aborts on device blas errors
    #define device_blas_check( status )  \
        do { \
            device_blas_status_t s = status; \
            blas::internal::abort_if( blas::is_device_error(s), __func__, \
                                      "%s", blas::device_error_string(s) ); \
        } while(0)

#else

    // blaspp throws device errors (default)
    #define device_error_check( error ) \
        do { \
            device_error_t e = error; \
            blas::internal::throw_if( blas::is_device_error(e), \
                                      blas::device_error_string(e), __func__ ); \
        } while(0)

    // blaspp throws device blas errors (default)
    #define device_blas_check( status ) \
        do { \
            device_blas_status_t s = status; \
            blas::internal::throw_if( blas::is_device_error(s), \
                                      blas::device_error_string(s), __func__ ); \
        } while(0)

#endif

// -----------------------------------------------------------------------------
// set/get device functions
void set_device(blas::Device device);
void get_device(blas::Device *device);

// -----------------------------------------------------------------------------
// conversion functions
device_trans_t   device_trans_const(blas::Op trans);
device_diag_t    device_diag_const(blas::Diag diag);
device_uplo_t    device_uplo_const(blas::Uplo uplo);
device_side_t    device_side_const(blas::Side side);

void device_free(void* ptr);
void device_free_pinned(void* ptr);

// -----------------------------------------------------------------------------
// Template functions declared here
// -----------------------------------------------------------------------------

/// @return a device pointer to an allocated memory space
template<typename T>
T* device_malloc(int64_t nelements)
{
    T* ptr = NULL;
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaMalloc((void**)&ptr, nelements * sizeof(T)) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: allocation for AMD GPUs
    #endif
    return ptr;
}

/// @return a host pointer to a pinned memory space
template<typename T>
T* device_malloc_pinned(int64_t nelements)
{
    T* ptr = NULL;
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaMallocHost((void**)&ptr, nelements * sizeof(T)) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: allocation using AMD driver API
    #endif
    return ptr;
}


// device set matrix
template<typename T>
void device_setmatrix(int64_t m, int64_t n,
                      T* hostPtr, int64_t ldh,
                      T* devPtr, int64_t ldd, Queue &queue)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasSetMatrixAsync(
                       (device_blas_int)m,    (device_blas_int)n, (device_blas_int)sizeof(T),
                       (const void *)hostPtr, (device_blas_int)ldh,
                       (      void *)devPtr,  (device_blas_int)ldd, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_set_matrix
    #endif
}

// device get matrix
template<typename T>
void device_getmatrix(int64_t m, int64_t n,
                      T* devPtr, int64_t ldd,
                      T* hostPtr, int64_t ldh, Queue &queue)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasGetMatrixAsync(
                       (device_blas_int)m,    (device_blas_int)n, (device_blas_int)sizeof(T),
                       (const void *)devPtr,  (device_blas_int)ldd,
                       (      void *)hostPtr, (device_blas_int)ldh, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_get_matrix
    #endif
}

// device set vector
template<typename T>
void device_setvector(int64_t n,
                      T* hostPtr, int64_t inch,
                      T* devPtr, int64_t incd, Queue &queue)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasSetVectorAsync(
                       (device_blas_int)n,    (device_blas_int)sizeof(T),
                       (const void *)hostPtr, (device_blas_int)inch,
                       (      void *)devPtr,  (device_blas_int)incd, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_set_vector
    #endif
}

// device get vector
template<typename T>
void device_getvector(int64_t n,
                      T* devPtr, int64_t incd,
                      T* hostPtr, int64_t inch, Queue &queue)
{
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasGetVectorAsync(
                       (device_blas_int)n,    (device_blas_int)sizeof(T),
                       (const void *)devPtr,  (device_blas_int)incd,
                       (      void *)hostPtr, (device_blas_int)inch, queue.stream() ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: call rocblas_get_vector
    #endif
}

}        //  namespace blas


#endif        //  #ifndef DEVICE_INTERNALS_HH
