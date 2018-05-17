#ifndef DEVICE_BATCH_TRSM_HH
#define DEVICE_BATCH_TRSM_HH

#include "device.hh"
#include <limits>
#include <cstring>

namespace blas {

namespace batch{
// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m, 
    std::vector<int64_t>    const &n, 
    std::vector<float >     const &alpha,
    std::vector<float*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<float*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                    std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::trsm_check<float>( side, uplo, trans, diag,  
                                        m, n, 
                                        alpha, Aarray, ldda, 
                                               Barray, lddb, 
                                        batch, info );
    }


    blas_error_if(false == is_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch));
    
    blas::set_device( queue.device() );
    if(is_vendor_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch)) {
        // call the vendor routine
        device_side_t   side_   = blas::device_side_const( side[0] );
        device_uplo_t   uplo_   = blas::device_uplo_const( uplo[0] );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        // copy Aarray, Barray, and Carray to device 
        float **dAarray, **dBarray;
        dAarray = (float**)queue.devPtrArray;
        dBarray = dAarray + batch;
        device_setvector<float*>(batch, (float**)&Aarray[0], 1, dAarray, 1, queue);
        device_setvector<float*>(batch, (float**)&Barray[0], 1, dBarray, 1, queue);
        DEVICE_BATCH_strsm( queue.handle(), 
                            side_, uplo_, trans_, diag_, 
                            m_, n_, alpha[0], 
                            dAarray, ldda_, 
                            dBarray, lddb_, batch);
    }
    else{
         // TODO: provide a reference implementation using queues
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m, 
    std::vector<int64_t>    const &n, 
    std::vector<double >     const &alpha,
    std::vector<double*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<double*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::trsm_check<double>( side, uplo, trans, diag,  
                                        m, n, 
                                        alpha, Aarray, ldda, 
                                               Barray, lddb, 
                                        batch, info );
    }


    blas_error_if(false == is_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch));
    
    blas::set_device( queue.device() );
    if(is_vendor_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch)) {
        // call the vendor routine
        device_side_t   side_   = blas::device_side_const( side[0] );
        device_uplo_t   uplo_   = blas::device_uplo_const( uplo[0] );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        // copy Aarray, Barray, and Carray to device 
        double **dAarray, **dBarray;
        dAarray = (double**)queue.devPtrArray;
        dBarray = dAarray + batch;
        device_setvector<double*>(batch, (double**)&Aarray[0], 1, dAarray, 1, queue);
        device_setvector<double*>(batch, (double**)&Barray[0], 1, dBarray, 1, queue);
        DEVICE_BATCH_dtrsm( queue.handle(), 
                            side_, uplo_, trans_, diag_, 
                            m_, n_, alpha[0], 
                            dAarray, ldda_, 
                            dBarray, lddb_, batch);
    }
    else{
         // TODO: provide a reference implementation using queues
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m, 
    std::vector<int64_t>    const &n, 
    std::vector<std::complex<float> >     const &alpha,
    std::vector<std::complex<float>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<float>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::trsm_check<std::complex<float>>( side, uplo, trans, diag,  
                                        m, n, 
                                        alpha, Aarray, ldda, 
                                               Barray, lddb, 
                                        batch, info );
    }


    blas_error_if(false == is_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch));
    
    blas::set_device( queue.device() );
    if(is_vendor_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch)) {
        // call the vendor routine
        device_side_t   side_   = blas::device_side_const( side[0] );
        device_uplo_t   uplo_   = blas::device_uplo_const( uplo[0] );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        // copy Aarray, Barray, and Carray to device 
        std::complex<float> **dAarray, **dBarray;
        dAarray = (std::complex<float>**)queue.devPtrArray;
        dBarray = dAarray + batch;
        device_setvector<std::complex<float>*>(batch, (std::complex<float>**)&Aarray[0], 1, dAarray, 1, queue);
        device_setvector<std::complex<float>*>(batch, (std::complex<float>**)&Barray[0], 1, dBarray, 1, queue);
        DEVICE_BATCH_ctrsm( queue.handle(), 
                            side_, uplo_, trans_, diag_, 
                            m_, n_, alpha[0], 
                            dAarray, ldda_, 
                            dBarray, lddb_, batch);
    }
    else{
         // TODO: provide a reference implementation using queues
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
inline
void trsm(
    std::vector<blas::Side> const &side,
    std::vector<blas::Uplo> const &uplo,
    std::vector<blas::Op>   const &trans,
    std::vector<blas::Diag> const &diag,
    std::vector<int64_t>    const &m, 
    std::vector<int64_t>    const &n, 
    std::vector<std::complex<double> >     const &alpha,
    std::vector<std::complex<double>*>     const &Aarray, std::vector<int64_t> const &ldda,
    std::vector<std::complex<double>*>     const &Barray, std::vector<int64_t> const &lddb,
    const size_t batch,                     std::vector<int64_t>       &info, 
    blas::Queue &queue )
{
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if(info.size() > 0){
        // perform error checking
        blas::batch::trsm_check<std::complex<double>>( side, uplo, trans, diag,  
                                        m, n, 
                                        alpha, Aarray, ldda, 
                                               Barray, lddb, 
                                        batch, info );
    }


    blas_error_if(false == is_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch));
    
    blas::set_device( queue.device() );
    if(is_vendor_supported("device", side, uplo, trans, diag, m, n, alpha, Aarray, ldda, Barray, lddb, batch)) {
        // call the vendor routine
        device_side_t   side_   = blas::device_side_const( side[0] );
        device_uplo_t   uplo_   = blas::device_uplo_const( uplo[0] );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        // copy Aarray, Barray, and Carray to device 
        std::complex<double> **dAarray, **dBarray;
        dAarray = (std::complex<double>**)queue.devPtrArray;
        dBarray = dAarray + batch;
        device_setvector<std::complex<double>*>(batch, (std::complex<double>**)&Aarray[0], 1, dAarray, 1, queue);
        device_setvector<std::complex<double>*>(batch, (std::complex<double>**)&Barray[0], 1, dBarray, 1, queue);
        DEVICE_BATCH_ztrsm( queue.handle(), 
                            side_, uplo_, trans_, diag_, 
                            m_, n_, alpha[0], 
                            dAarray, ldda_, 
                            dBarray, lddb_, batch);
    }
    else{
         // TODO: provide a reference implementation using queues
    }
}


}        //  namespace batch
}        //  namespace blas

#endif        //  #ifndef DEVICE_BATCH_TRSM_HH

