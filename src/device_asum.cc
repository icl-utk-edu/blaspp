#include "blas/device_blas.hh"
#include "device_internal.hh"

#include <limits>

namespace blas {

//==============================================================================
namespace impl {

template <typename scalar_t, typename scalar_o>
void asum(
    int64_t n,
    scalar_t const *x, int64_t incx,
    scalar_o *result,
    blas::Queue &queue )
{
#ifndef BLAS_HAVE_DEVICE
    throw blas::Error( "device BLAS not available", __func__ );
#else
    blas_error_if(n < 0);
    blas_error_if(incx == 0);
    //check param
    device_blas_int n_ = to_device_blas_int(n);
    device_blas_int incx_ = to_device_blas_int(incx);

    blas::internal_set_device( queue.device() );
    #if defined( BLAS_HAVE_SYCL )
        return;
    #else
        internal::asum(n_, x, incx_, result, queue);
    #endif
#endif
}

}//namespace impl
//---------------------------------------------------------
///@ingroup asum
void asum(    
    int64_t n,
    float const *x, int64_t incdx,
    float *result, 
    blas::Queue& queue ){
        //printf("asum  routine 1\n");
        impl::asum(n, x, incdx, result, queue);
    }
//---------------------------------------------------------
///@ingroup asum
void asum(    
    int64_t n,
    double const *x, int64_t incdx,
    double *result, 
    blas::Queue& queue ){
        //printf("asum  routine 2\n");
        impl::asum(n, x, incdx, result, queue);
    }
//---------------------------------------------------------
///@ingroup asum
void asum(    
    int64_t n,
    std::complex<float> const *x, int64_t incdx,
    float *result, 
    blas::Queue& queue ){
        //printf("asum  routine 3\n");
        impl::asum(n, x, incdx, result, queue);
    }
//---------------------------------------------------------
///@ingroup asum
void asum(    
    int64_t n,
    std::complex<double> const *x, int64_t incdx,
    double *result, 
    blas::Queue& queue ){
        //printf("asum  routine 4\n");
        impl::asum(n, x, incdx, result, queue);
    }


}//namespace blas