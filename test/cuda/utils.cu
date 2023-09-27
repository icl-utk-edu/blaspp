#include "../utils.cuh"

//------------------------------------------------------------------------------
/// @return ceil( x / y ), for integer type T.
template <typename T>
inline constexpr T ceildiv( T x, T y )
{
    return T( (x + y - 1) / y );
}

//==============================================================================
// Overloads to enable templating.

//------------------------------------------------------------------------------
// Template implementation when C++ has default rules for conversion
// (or no conversion needed).
template <typename src_t, typename dst_t>
__host__ __device__
inline void copy_scalar( src_t src, dst_t& dst )
{
    dst = dst_t( src );
}

//------------------------------------------------------------------------------
// Overloaded implementations for specific cases.
__host__ __device__
inline void copy_scalar( float src, __half& dst )
{
    dst = __float2half( src );
}

//------------------------------------------------------------------------------
__host__ __device__
inline void copy_scalar( double src, __half& dst )
{
    dst = __double2half( src );
}

//------------------------------------------------------------------------------
__host__ __device__
inline void copy_scalar( __half src, float& dst )
{
    dst = __half2float( src );
}

//------------------------------------------------------------------------------
__host__ __device__
inline void copy_scalar( __half src, double& dst )
{
    // no __half2double, so do in 2 steps
    dst = double( __half2float( src ) );
}

//==============================================================================
// GPU function.
const int blk_x = 32;
const int blk_y = 4;

//------------------------------------------------------------------------------
// GPU device routine, called from GPU kernel.
//
// Each thread-block does a blk_x by blk_y block of the matrix;
// each thread does 1 entry in the block.
// Because of the CUDA max y grid dimension of 65535, the entire grid is
// repeated in the y dimension with step = gridDim.y * blockDim.y,
// so thread (i, j) will do entries (i, j), (i, j + step), (i, j + 2*step), ...
// The max x grid dimension is 2^31-1, so there's no need to repeat.
// Cf. magma/magmablas/slag2h.cu
//
template <typename src_t, typename dst_t>
__device__
void copy_matrix_device(
    int m, int n,
    src_t const* src, int ld_src,
    dst_t*       dst, int ld_dst )
{
    // Global thread index.
    const int ti = blockIdx.x * blockDim.x + threadIdx.x;
    const int tj = blockIdx.y * blockDim.y + threadIdx.y;

    if (ti < m) {
        for (int j = tj; j < n; j += gridDim.y * blockDim.y) {
            copy_scalar( src[ ti + j*ld_src ],
                         dst[ ti + j*ld_dst ] );
        }
    }
}

//------------------------------------------------------------------------------
// GPU kernel, called from CPU driver.
template <typename src_t, typename dst_t>
__global__
void copy_matrix_kernel(
    int m, int n,
    src_t const* src, int ld_src,
    dst_t*       dst, int ld_dst )
{
    copy_matrix_device( m, n, src, ld_src, dst, ld_dst );
}

namespace blas {

//------------------------------------------------------------------------------
// Copy m-by-n src matrix to dst matrix, with type conversion.
template <typename src_t, typename dst_t>
void copy_matrix(
    int m, int n,
    src_t const* src, int ld_src,
    dst_t*       dst, int ld_dst,
    blas::Queue &queue)
{

    cudaStream_t stream = queue.stream();

    // CUDA has max x grid dimension of 2^31-1; y, z grid dimensions of 65535.
    dim3 threads( blk_x, blk_y );
    dim3 blocks( ceildiv( m, blk_x ), std::min( 4, ceildiv( n, blk_y ) ) );

    // printf( "%s: m %d, n %d; threads %d, %d, %d; blocks %d, %d, %d\n",
    //         __func__,
    //         threads.x, threads.y, threads.z,
    //         blocks.x,  blocks.y,  blocks.z );

    copy_matrix_kernel<<< blocks, threads, 0, stream >>>
        ( m, n, src, ld_src, dst, ld_dst );

    // Check that launch succeeded. This doesn't say execution will be
    // successful, it checks only the launch.
    blas_dev_call(
        cudaGetLastError() );
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void copy_matrix<float, half>(
    int m, int n,
    float const* src, int ld_src,
    half*        dst, int ld_dst,
    blas::Queue &queue);

template
void copy_matrix<half, float>(
    int m, int n,
    half const* src, int ld_src,
    float*      dst, int ld_dst,
    blas::Queue &queue);

template
void copy_matrix<float, float>(
    int m, int n,
    float const* src, int ld_src,
    float*       dst, int ld_dst,
    blas::Queue &queue);
} // namespace blas
