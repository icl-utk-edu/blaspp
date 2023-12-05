#include "utils.hh"

namespace blas {

template <typename src_t, typename dst_t>
void copy_matrix(
    int m, int n,
    src_t const* src, int ld_src,
    dst_t*       dst, int ld_dst)
{
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      dst[i + j*ld_dst] = (dst_t)src[i + j*ld_src];
    }
  }
}

//------------------------------------------------------------------------------
// Explicit instantiations.
template
void copy_matrix<float, float16>(
    int m, int n,
    float const* src, int ld_src,
    float16*     dst, int ld_dst);

template
void copy_matrix<float16, float>(
    int m, int n,
    float16 const* src, int ld_src,
    float*         dst, int ld_dst);

} // namespace blas
