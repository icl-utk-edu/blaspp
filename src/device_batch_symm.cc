#include <limits>
#include <cstring>
#include "batch_common.hh"
#include "device_blas.hh"

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<int64_t>    const &m, 
    std::vector<int64_t>    const &n, 
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<float >     const &beta,
    std::vector<float*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::symm_check<float>( 
                        side, uplo, 
                        m, n, 
                        alpha, Aarray, ldda, 
                               Barray, lddb, 
                        beta,  Carray, lddc, 
                        batch, info );
    }

    throw std::exception();  // not yet available
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m, 
    std::vector<int64_t>     const &n, 
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<double >     const &beta,
    std::vector<double*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::symm_check<double>( 
                        side, uplo, 
                        m, n, 
                        alpha, Aarray, ldda, 
                               Barray, lddb, 
                        beta,  Carray, lddc, 
                        batch, info );
    }

    throw std::exception();  // not yet available by the vendor
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m, 
    std::vector<int64_t>     const &n, 
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<float> >     const &beta,
    std::vector<std::complex<float>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::symm_check<std::complex<float>>( 
                        side, uplo, 
                        m, n, 
                        alpha, Aarray, ldda, 
                               Barray, lddb, 
                        beta,  Carray, lddc, 
                        batch, info );
    }

    throw std::exception();  // not yet available by the vendor
}

// -----------------------------------------------------------------------------
/// @ingroup symm
void blas::batch::symm(
    std::vector<blas::Side>  const &side,
    std::vector<blas::Uplo>  const &uplo,
    std::vector<int64_t>     const &m, 
    std::vector<int64_t>     const &n, 
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    std::vector<std::complex<double> >     const &beta,
    std::vector<std::complex<double>*>     const &Carray, std::vector<int64_t> const &lddc,
    const size_t batch,                    std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::symm_check<std::complex<double>>( 
                        side, uplo, 
                        m, n, 
                        alpha, Aarray, ldda, 
                               Barray, lddb, 
                        beta,  Carray, lddc, 
                        batch, info );
    }

    throw std::exception();  // not yet available by the vendor
}
