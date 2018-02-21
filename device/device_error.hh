#ifndef DEVICE_ERROR_HH
#define DEVICE_ERROR_HH

namespace blas{

// ----------------------------------------------------------------------------- 
inline 
bool is_device_error(device_error_t error){ return (error != DevSuccess); }

inline
bool is_device_error(device_blas_status_t status){ return (status != DevBlasSuccess); }

// ----------------------------------------------------------------------------- 
inline 
const char* device_error_string(device_error_t error){
    #ifdef HAVE_CUBLAS
    return cudaGetErrorString( error );
    #elif defined(HAVE_ROCBLAS)
    // TODO: return error string for rocblas
    #endif
}

inline 
const char* device_error_string(device_blas_status_t status){
    switch( status ) {
    #ifdef HAVE_CUBLAS
        case CUBLAS_STATUS_SUCCESS:
            return "device blas: success";

        case CUBLAS_STATUS_NOT_INITIALIZED:
            return "device blas: not initialized";

        case CUBLAS_STATUS_ALLOC_FAILED:
            return "device blas: out of memory";

        case CUBLAS_STATUS_INVALID_VALUE:
            return "device blas: invalid value";

        case CUBLAS_STATUS_ARCH_MISMATCH:
            return "device blas: architecture mismatch";

        case CUBLAS_STATUS_MAPPING_ERROR:
            return "device blas: memory mapping error";

        case CUBLAS_STATUS_EXECUTION_FAILED:
            return "device blas: execution failed";

        case CUBLAS_STATUS_INTERNAL_ERROR:
            return "device blas: internal error";

        case CUBLAS_STATUS_NOT_SUPPORTED:
            return "device blas: functionality not supported";

        case CUBLAS_STATUS_LICENSE_ERROR:
            return "device blas: license error";
    #elif defined(HAVE_ROCBLAS)
    // TODO: return error string for rocblas
    #endif
        default:
            return "unknown device blas error code";
    }
}

// -----------------------------------------------------------------------------
// internal macros to handle device errors 
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
        do{ device_error_t e = error; \
            blas::internal::abort_if( blas::is_device_error(e), __func__, "%s", blas::device_error_string(e) ); } while(0)

    // blaspp aborts on device blas errors
    #define device_blas_check( status )  \
        do{ device_blas_status_t s = status; \
            blas::internal::abort_if( blas::is_device_error(s), __func__, "%s", blas::device_error_string(s) ); } while(0)

#else

    // blaspp throws device errors (default)
    #define device_error_check( error ) \
        do{ device_error_t e = error; \
            blas::internal::throw_if( blas::is_device_error(e), blas::device_error_string(e), __func__ ); } while(0)

    // blaspp throws device blas errors (default)
    #define device_blas_check( status ) \
        do{ device_blas_status_t s = status; \
            blas::internal::throw_if( blas::is_device_error(s), blas::device_error_string(s), __func__ ); } while(0)

#endif

}    // namespace blas


#endif        //  #ifndef DEVICE_ERROR_HH

