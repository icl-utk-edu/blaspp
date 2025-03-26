#include "blas/device.hh"

#if defined(BLAS_HAVE_CUBLAS)

namespace blas {

template <typename scalar_t>
__global__ void shift_vec_kernel(
    int64_t n, scalar_t* v,
    int64_t c)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v[ i ] += c;
    }
}

template <typename scalar_t>
void cuda_shift_vec(
    int64_t n, scalar_t* v,
    int64_t c,
    blas::Queue &queue)
{
    if (n == 0) {
        return;
    }

    int64_t nthreads = std::min( int64_t( 1024 ), n );

    cudaSetDevice( queue.device() );

    shift_vec_kernel<<<1, nthreads, 0, queue.stream()>>>( n, v, c );

    cudaError_t error = cudaGetLastError();
    assert ( error == cudaSuccess );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void cuda_shift_vec(
    int64_t n, int64_t* v,
    int64_t c,
    blas::Queue &queue);

template
    void cuda_shift_vec(
        int64_t n, int* v,
        int64_t c,
        blas::Queue &queue);

} // namespace blas

#endif // BLAS_HAVE_CUBLAS