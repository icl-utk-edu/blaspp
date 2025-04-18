#include "hip/hip_runtime.h"
#include "blas/device.hh"

#if defined(BLAS_HAVE_ROCBLAS)

namespace blas {

template <typename scalar_t>
__device__ void shift_vec_func(
    int64_t n, scalar_t* v,
    scalar_t c)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v[ i ] += c;
    }
}

template <typename scalar_t>
__global__ void shift_vec_kernel(
    int64_t n, scalar_t* v,
    scalar_t c)
{
    shift_vec_func( n, v, c );
}

template <typename scalar_t>
void shift_vec(
    int64_t n, scalar_t* v,
    scalar_t c,
    blas::Queue& queue)
{
    if (n == 0) {
        return;
    }

    int64_t nthreads = std::min( int64_t( 1024 ), n );

    blas_dev_call(
        hipSetDevice( queue.device() ) );

    shift_vec_kernel<<<1, nthreads, 0, queue.stream()>>>(
        n, v, c);

    blas_dev_call(
        hipGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void shift_vec(
    int64_t n, int64_t* v,
    int64_t c,
    blas::Queue& queue);

} // namespace blas

#endif // BLAS_HAVE_ROCBLAS
