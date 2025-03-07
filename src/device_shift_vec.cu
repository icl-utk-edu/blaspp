#include "blas/device.hh"

#if defined(BLAS_HAVE_CUBLAS)

namespace blas {

template <typename scalar_t>
__global__ void shift_vec_kernel(
    int64_t n, scalar_t* v,
    int64_t c)
{
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        v[i] += c;
    }
}

template <typename scalar_t>
void shift_vec(
    int64_t n, scalar_t* v,
    int64_t c,
    int64_t batch_count, blas::Queue &queue)
{
    if (batch_count == 0) {
        return;
    }

    int64_t nthreads = std::min( int64_t( 1024 ), n );

    cudaSetDevice( queue.device() );

    shift_vec_kernel<<<batch_count, nthreads, 0, queue.stream()>>>(
        n, v, c);

    cudaError_t error = cudaGetLastError();
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void shift_vec(
    int64_t n, int64_t* v,
    int64_t c,
    int64_t batch_count, blas::Queue &queue);

} // namespace blas

#endif // BLAS_HAVE_CUBLAS