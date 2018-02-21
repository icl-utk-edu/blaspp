#ifndef DEVICE_QUEUE_HH
#define DEVICE_QUEUE_HH

namespace blas {

class Queue
{
public:
    Queue(){
        blas::get_device( &device_ );
        #ifdef HAVE_CUBLAS
        device_error_check( cudaStreamCreate(&stream_) );
        device_blas_check( cublasCreate(&handle_) );
        device_blas_check( cublasSetStream( handle_, stream_ ) );
        #elif defined(HAVE_ROCBLAS)
        // TODO: rocBLAS queue init
        #endif
    }

    Queue(blas::Device device, int64_t batch_size){
        device_ = device;
        blas::set_device( device_ );
        #ifdef HAVE_CUBLAS
        device_error_check( cudaStreamCreate(&stream_) );
        device_blas_check( cublasCreate(&handle_) );
        device_blas_check( cublasSetStream( handle_, stream_ ) );
        Aarray.resize(batch_size);
        Barray.resize(batch_size);
        Carray.resize(batch_size);
        #elif defined(HAVE_ROCBLAS)
        // TODO: rocBLAS queue init and vector resize
        #endif
    }

    /// @return device associated with this queue
    blas::Device   device()          { return device_;   }

    /// @return device blas handle associated with this queue
    device_blas_handle_t   handle()   { return handle_;   }
    
    #ifdef HAVE_CUBLAS
    /// @return CUDA stream associated with this queue; requires CUDA.
    cudaStream_t     stream()     { return stream_;   }
    #elif defined(HAVE_ROCBLAS)
    // TODO: add similar functionality for rocBLAS, if required
    #endif

    
    /// synchronize with queue.
    void sync(){
        #ifdef HAVE_CUBLAS
        device_error_check( cudaStreamSynchronize(this->stream()) );
        #elif defined(HAVE_ROCBLAS)
        // TODO: sync with queue in rocBLAS
        #endif
    }
    
    ~Queue(){
        #ifdef HAVE_CUBLAS
        device_blas_check( cublasDestroy(handle_) );
        device_error_check( cudaStreamDestroy(stream_) );
        Aarray.clear();
        Barray.clear();
        Carray.clear();
        #elif defined(HAVE_ROCBLAS)
        // TODO: rocBLAS equivalent
        #endif
    }

    
private:
    blas::Device   device_;      // associated device ID

    device_blas_handle_t   handle_;      // associated device blas handle

    #ifdef HAVE_CUBLAS
    cudaStream_t     stream_;      // associated CUDA stream; may be NULL
    // pointer arrays for batch routines
    // precision-independent
    thrust::device_vector<void*> Aarray;
    thrust::device_vector<void*> Barray;
    thrust::device_vector<void*> Carray;
    #elif defined(HAVE_ROCBLAS)
    // TODO: stream and pointer arrays for rocBLAS
    #endif
    
};

}        //  namespace blas

#endif        //  #ifndef DEVICE_QUEUE_HH

