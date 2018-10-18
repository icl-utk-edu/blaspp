#include "device.hh"

/** queue member functions **/

// -----------------------------------------------------------------------------
// default constructor 
blas::Queue::Queue()
{
    blas::get_device( &device_ );
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaStreamCreate(&stream_) );
    device_blas_check( cublasCreate(&handle_) );
    device_blas_check( cublasSetStream( handle_, stream_ ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS queue init
    #endif
}

// -----------------------------------------------------------------------------
// constructor with batch init 
blas::Queue::Queue(blas::Device device, int64_t batch_size)
{
    device_ = device;
    blas::set_device( device_ );
    hostPtrArray = blas::device_malloc_pinned<void*>( 3 * batch_size );
    devPtrArray  = blas::device_malloc<void*>( 3 * batch_size );
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaStreamCreate(&stream_) );
    device_blas_check( cublasCreate(&handle_) );
    device_blas_check( cublasSetStream( handle_, stream_ ) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS queue init and vector resize
    #endif
}

// -----------------------------------------------------------------------------
/// @return device associated with this queue
blas::Device   blas::Queue::device()          { return device_;   }

// -----------------------------------------------------------------------------
/// @return device blas handle associated with this queue
device_blas_handle_t   blas::Queue::handle()   { return handle_;   }
    
// -----------------------------------------------------------------------------
#ifdef BLASPP_WITH_CUBLAS
/// @return CUDA stream associated with this queue; requires CUDA.
cudaStream_t     blas::Queue::stream()     { return stream_;   }
#elif defined(HAVE_ROCBLAS)
// TODO: add similar functionality for rocBLAS, if required
#endif

    
// -----------------------------------------------------------------------------
/// synchronize with queue.
void blas::Queue::sync()
{
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaStreamSynchronize(this->stream()) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: sync with queue in rocBLAS
    #endif
}
    
// -----------------------------------------------------------------------------
// default destructor
blas::Queue::~Queue()
{
    blas::device_free( devPtrArray );
    blas::device_free_pinned( hostPtrArray );
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasDestroy(handle_) );
    device_error_check( cudaStreamDestroy(stream_) );
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif
}
