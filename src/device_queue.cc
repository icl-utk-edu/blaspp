#include "device.hh"

/** queue member functions **/

// -----------------------------------------------------------------------------
// default constructor
blas::Queue::Queue()
{
    // get the currently set device ID
    blas::get_device( &device_ );
    batch_limit_ = DEV_QUEUE_DEFAULT_BATCH_LIMIT; 
    // compute workspace for pointer arrays in the queue
    // fork size + 1 (def. stream), each need 3 pointer arrays
    size_t workspace_size = 3 * batch_limit_ * ( DEV_QUEUE_FORK_SIZE + 1 );
    devPtrArray  = blas::device_malloc<void*>( workspace_size );
    #ifdef BLASPP_WITH_CUBLAS
    // default stream
    device_error_check( cudaStreamCreate(&default_stream_) );
    device_blas_check( cublasCreate(&handle_) );
    device_blas_check( cublasSetStream( handle_, default_stream_ ) );
    current_stream_       = &default_stream_;
    num_active_streams_   = 1;
    current_stream_index_ = 0; 

    // create parallel streams
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaStreamCreate(&parallel_streams_[ i ]) );
    }

    // create default and parallel events
    device_error_check( cudaEventCreate(&default_event_) );
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaEventCreate(&parallel_events_[ i ]) );
    }
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS queue init
    #endif
}

// -----------------------------------------------------------------------------
// constructor with batch init
blas::Queue::Queue(blas::Device device, int64_t batch_size)
{
    device_ = device;
    batch_limit_ = batch_size; 
    blas::set_device( device_ );
    // compute workspace for pointer arrays in the queue
    // fork size + 1 (def. stream), each need 3 pointer arrays
    size_t workspace_size = 3 * batch_limit_ * ( DEV_QUEUE_FORK_SIZE + 1 );
    devPtrArray  = blas::device_malloc<void*>( workspace_size );
    #ifdef BLASPP_WITH_CUBLAS
    device_error_check( cudaStreamCreate(&default_stream_) );
    device_blas_check( cublasCreate(&handle_) );
    device_blas_check( cublasSetStream( handle_, default_stream_ ) );
    current_stream_       = &default_stream_;
    num_active_streams_   = 1;
    current_stream_index_ = 0; 

    // create parallel streams
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaStreamCreate(&parallel_streams_[ i ]) );
    }

    // create default and parallel events
    device_error_check( cudaEventCreate(&default_event_) );
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaEventCreate(&parallel_events_[ i ]) );
    }
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
cudaStream_t     blas::Queue::stream()     { return *current_stream_;   }
#elif defined(HAVE_ROCBLAS)
// TODO: add similar functionality for rocBLAS, if required
#endif


// -----------------------------------------------------------------------------
/// synchronize with queue.
void blas::Queue::sync()
{
    #ifdef BLASPP_WITH_CUBLAS
    // in default mode, sync with default stream
    // otherwise, sync against the parallel streams
    if( current_stream_ == &default_stream_ ) {
        device_error_check( cudaStreamSynchronize( default_stream_ ) );
    }
    else {
        for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
            device_error_check( cudaStreamSynchronize(parallel_streams_[ i ]) );
        }
    }
    #elif defined(HAVE_ROCBLAS)
    // TODO: sync with queue in rocBLAS
    #endif
}

// -----------------------------------------------------------------------------
/// get batch limit.
size_t blas::Queue::get_batch_limit()    { return batch_limit_; }

// -----------------------------------------------------------------------------
/// get device array pointer for the current stream.
void**  blas::Queue::get_devPtrArray()    
{ 
    // in default (join) mode, return devPtrArray with no offset
    if( current_stream_ == &default_stream_ ) return devPtrArray;

    // in fork mode, return devPtrArray + offset
    size_t offset = ( current_stream_index_ + 1 ) * 3 * batch_limit_;
    return (devPtrArray + offset);
}

// -----------------------------------------------------------------------------
/// forks the kernel launches assigned to this queue to parallel streams
/// this function is not nested ( you must join after each fork )
void blas::Queue::fork()
{
    #ifdef BLASPP_WITH_CUBLAS
    // check if queue is already in fork mode
    if( current_stream_ != &default_stream_ ) return;

    // make sure dependencies are respected
    device_error_check( cudaEventRecord(default_event_, default_stream_) );
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaStreamWaitEvent(parallel_streams_[i], default_event_, 0) );
    }

    // assign current stream
    current_stream_index_ = 0;
    num_active_streams_   = DEV_QUEUE_FORK_SIZE;
    current_stream_       = &parallel_streams_[ current_stream_index_ ];

    // assign cublas handle to current stream
    device_blas_check( cublasSetStream( handle_, *current_stream_ ) );
    #else
    // TODO: rocBLAS equivalent
    #endif
}

// -----------------------------------------------------------------------------
/// switch back executions on this queue from parallel streams to the default stream
/// this function is not nested ( you must join after each fork )
void blas::Queue::join()
{
    #ifdef BLASPP_WITH_CUBLAS
    // check if queue is already joined
    if( current_stream_ == &default_stream_ ) return;

    // make sure dependencies are respected
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaEventRecord(parallel_events_[i], parallel_streams_[i]) );
        device_error_check( cudaStreamWaitEvent(default_stream_, parallel_events_[i], 0) );
    }

    // assign current stream
    current_stream_index_ = 0;
    num_active_streams_   = 1;
    current_stream_       = &default_stream_;

    // assign cublas handle to current stream
    device_blas_check( cublasSetStream( handle_, *current_stream_ ) );
    #else
    // TODO: rocBLAS equivalent
    #endif
}

// -----------------------------------------------------------------------------
/// in fork mode, switch execution to the next-in-line stream
/// in join mode, no effect
void blas::Queue::revolve()
{
    #ifdef BLASPP_WITH_CUBLAS
    // return if not in fork mode
    if( current_stream_ == &default_stream_ ) return;
    
    // choose the next-in-line stream
    current_stream_index_ = (current_stream_index_ + 1) % num_active_streams_;
    current_stream_       = &parallel_streams_[ current_stream_index_ ];

    // assign cublas handle to current stream
    device_blas_check( cublasSetStream( handle_, *current_stream_ ) );
    #else
    // TODO: rocBLAS equivalent
    #endif
}

// -----------------------------------------------------------------------------
// default destructor
blas::Queue::~Queue()
{
    blas::device_free( devPtrArray );
    #ifdef BLASPP_WITH_CUBLAS
    device_blas_check( cublasDestroy(handle_) );
    device_error_check( cudaStreamDestroy(default_stream_) );

    // destroy parallel streams
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check(cudaStreamDestroy( parallel_streams_[ i ] ));
    }

    // destroy events
    device_error_check( cudaEventDestroy(default_event_) );
    for(size_t i = 0; i < DEV_QUEUE_FORK_SIZE; i++) {
        device_error_check( cudaEventDestroy(parallel_events_[ i ]) );
    }
    #elif defined(HAVE_ROCBLAS)
    // TODO: rocBLAS equivalent
    #endif
}
