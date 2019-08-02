#include <limits>
#include <cstring>
#include "batch_common.hh"
#include "device_blas.hh"

// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::batch::trsm(
    blas::Layout                   layout,
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
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trsm_check<float>( layout, side, uplo, trans, diag,
                                        m, n,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        batch, info );
    }

    bool fixed_size =   ( side.size()   == 1     &&
                          uplo.size()   == 1     &&
                          trans.size()  == 1     &&
                          diag.size()   == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1  );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        blas::Uplo luplo = uplo[0]; // local value
        blas::Side lside = side[0]; // local value
        if (layout == Layout::RowMajor) {
            // swap lower <=> upper, left <=> right, m <=> n
            luplo = (luplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
            lside = (lside == Side::Left ? Side::Right : Side::Left);
            std::swap( m_, n_ );
        }

        device_side_t   side_   = blas::device_side_const( lside );
        device_uplo_t   uplo_   = blas::device_uplo_const( luplo );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );

        size_t batch_limit = queue.get_batch_limit();
        float **dAarray, **dBarray;
        dAarray = (float**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy pointer array(s) to device
            device_setvector<float*>(ibatch, (float**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector<float*>(ibatch, (float**)&Barray[ib], 1, dBarray, 1, queue);

            DEVICE_BATCH_strsm( queue.handle(),
                                side_, uplo_, trans_, diag_,
                                m_, n_, alpha[0],
                                dAarray, ldda_,
                                dBarray, lddb_, ibatch);
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Side side_   = blas::batch::extract<Side>(side, i);
            Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
            Op   trans_  = blas::batch::extract<Op>(trans, i);
            Diag diag_   = blas::batch::extract<Diag>(diag, i);
            int64_t m_   = blas::batch::extract<int64_t>(m, i);
            int64_t n_   = blas::batch::extract<int64_t>(n, i);
            int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
            float alpha_ = blas::batch::extract<float>(alpha, i);
            float* dA_   = blas::batch::extract<float*>(Aarray, i);
            float* dB_   = blas::batch::extract<float*>(Barray, i);
            blas::trsm(
                layout, side_, uplo_, trans_, diag_, m_, n_,
                alpha_, dA_, lda_,
                        dB_, ldb_, queue );
            queue.revolve();
        }
        queue.join();
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::batch::trsm(
    blas::Layout                   layout,
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
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trsm_check<double>( layout, side, uplo, trans, diag,
                                        m, n,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        batch, info );
    }

    bool fixed_size =   ( side.size()   == 1     &&
                          uplo.size()   == 1     &&
                          trans.size()  == 1     &&
                          diag.size()   == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1  );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        blas::Uplo luplo = uplo[0]; // local value
        blas::Side lside = side[0]; // local value
        if (layout == Layout::RowMajor) {
            // swap lower <=> upper, left <=> right, m <=> n
            luplo = (luplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
            lside = (lside == Side::Left ? Side::Right : Side::Left);
            std::swap( m_, n_ );
        }

        device_side_t   side_   = blas::device_side_const( lside );
        device_uplo_t   uplo_   = blas::device_uplo_const( luplo );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );

        size_t batch_limit = queue.get_batch_limit();
        double **dAarray, **dBarray;
        dAarray = (double**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy pointer array(s) to device
            device_setvector<double*>(ibatch, (double**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector<double*>(ibatch, (double**)&Barray[ib], 1, dBarray, 1, queue);

            DEVICE_BATCH_dtrsm( queue.handle(),
                                side_, uplo_, trans_, diag_,
                                m_, n_, alpha[0],
                                dAarray, ldda_,
                                dBarray, lddb_, ibatch);
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Side side_   = blas::batch::extract<Side>(side, i);
            Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
            Op   trans_  = blas::batch::extract<Op>(trans, i);
            Diag diag_   = blas::batch::extract<Diag>(diag, i);
            int64_t m_   = blas::batch::extract<int64_t>(m, i);
            int64_t n_   = blas::batch::extract<int64_t>(n, i);
            int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
            double alpha_ = blas::batch::extract<double>(alpha, i);
            double* dA_   = blas::batch::extract<double*>(Aarray, i);
            double* dB_   = blas::batch::extract<double*>(Barray, i);
            blas::trsm(
                layout, side_, uplo_, trans_, diag_, m_, n_,
                alpha_, dA_, lda_,
                        dB_, ldb_, queue );
            queue.revolve();
        }
        queue.join();
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::batch::trsm(
    blas::Layout                   layout,
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
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trsm_check<std::complex<float>>( layout, side, uplo, trans, diag,
                                        m, n,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        batch, info );
    }

    bool fixed_size =   ( side.size()   == 1     &&
                          uplo.size()   == 1     &&
                          trans.size()  == 1     &&
                          diag.size()   == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1  );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        blas::Uplo luplo = uplo[0]; // local value
        blas::Side lside = side[0]; // local value
        if (layout == Layout::RowMajor) {
            // swap lower <=> upper, left <=> right, m <=> n
            luplo = (luplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
            lside = (lside == Side::Left ? Side::Right : Side::Left);
            std::swap( m_, n_ );
        }

        device_side_t   side_   = blas::device_side_const( lside );
        device_uplo_t   uplo_   = blas::device_uplo_const( luplo );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );

        size_t batch_limit = queue.get_batch_limit();
        std::complex<float> **dAarray, **dBarray;
        dAarray = (std::complex<float>**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy pointer array(s) to device
            device_setvector< std::complex<float>* >(ibatch, (std::complex<float>**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector< std::complex<float>* >(ibatch, (std::complex<float>**)&Barray[ib], 1, dBarray, 1, queue);

            DEVICE_BATCH_ctrsm( queue.handle(),
                                side_, uplo_, trans_, diag_,
                                m_, n_, alpha[0],
                                dAarray, ldda_,
                                dBarray, lddb_, ibatch);
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Side side_   = blas::batch::extract<Side>(side, i);
            Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
            Op   trans_  = blas::batch::extract<Op>(trans, i);
            Diag diag_   = blas::batch::extract<Diag>(diag, i);
            int64_t m_   = blas::batch::extract<int64_t>(m, i);
            int64_t n_   = blas::batch::extract<int64_t>(n, i);
            int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
            std::complex<float> alpha_ = blas::batch::extract<std::complex<float> >(alpha, i);
            std::complex<float>* dA_   = blas::batch::extract<std::complex<float>*>(Aarray, i);
            std::complex<float>* dB_   = blas::batch::extract<std::complex<float>*>(Barray, i);
            blas::trsm(
                layout, side_, uplo_, trans_, diag_, m_, n_,
                alpha_, dA_, lda_,
                        dB_, ldb_, queue );
            queue.revolve();
        }
        queue.join();
    }
}


// -----------------------------------------------------------------------------
/// @ingroup trsm
void blas::batch::trsm(
    blas::Layout                   layout,
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
    blas_error_if( layout != Layout::ColMajor && layout != Layout::RowMajor );
    blas_error_if( batch < 0 );
    blas_error_if( !(info.size() == 0 || info.size() == 1 || info.size() == batch) );
    if (info.size() > 0) {
        // perform error checking
        blas::batch::trsm_check<std::complex<double>>( layout, side, uplo, trans, diag,
                                        m, n,
                                        alpha, Aarray, ldda,
                                               Barray, lddb,
                                        batch, info );
    }

    bool fixed_size =   ( side.size()   == 1     &&
                          uplo.size()   == 1     &&
                          trans.size()  == 1     &&
                          diag.size()   == 1     &&
                          m.size()      == 1     &&
                          n.size()      == 1     &&
                          alpha.size()  == 1     &&
                          Aarray.size() == batch &&
                          ldda.size()   == 1     &&
                          Barray.size() == batch &&
                          lddb.size()   == 1  );

    blas::set_device( queue.device() );
    if (fixed_size) {
        // call the vendor routine
        device_blas_int m_      = (device_blas_int) m[0];
        device_blas_int n_      = (device_blas_int) n[0];
        device_blas_int ldda_   = (device_blas_int) ldda[0];
        device_blas_int lddb_   = (device_blas_int) lddb[0];

        blas::Uplo luplo = uplo[0]; // local value
        blas::Side lside = side[0]; // local value
        if (layout == Layout::RowMajor) {
            // swap lower <=> upper, left <=> right, m <=> n
            luplo = (luplo == Uplo::Lower ? Uplo::Upper : Uplo::Lower);
            lside = (lside == Side::Left ? Side::Right : Side::Left);
            std::swap( m_, n_ );
        }

        device_side_t   side_   = blas::device_side_const( lside );
        device_uplo_t   uplo_   = blas::device_uplo_const( luplo );
        device_trans_t  trans_  = blas::device_trans_const( trans[0] );
        device_diag_t   diag_   = blas::device_diag_const( diag[0] );

        size_t batch_limit = queue.get_batch_limit();
        std::complex<double> **dAarray, **dBarray;
        dAarray = (std::complex<double>**)queue.get_devPtrArray();
        dBarray = dAarray + batch_limit;

        for( size_t ib = 0; ib < batch; ib += batch_limit ) { 
            size_t ibatch = std::min( batch_limit, batch-ib );

            // copy pointer array(s) to device
            device_setvector< std::complex<double>* >(ibatch, (std::complex<double>**)&Aarray[ib], 1, dAarray, 1, queue);
            device_setvector< std::complex<double>* >(ibatch, (std::complex<double>**)&Barray[ib], 1, dBarray, 1, queue);

            DEVICE_BATCH_ztrsm( queue.handle(),
                                side_, uplo_, trans_, diag_,
                                m_, n_, alpha[0],
                                dAarray, ldda_,
                                dBarray, lddb_, ibatch);
        }
    }
    else {
        queue.fork();
        for (size_t i = 0; i < batch; ++i) {
            Side side_   = blas::batch::extract<Side>(side, i);
            Uplo uplo_   = blas::batch::extract<Uplo>(uplo, i);
            Op   trans_  = blas::batch::extract<Op>(trans, i);
            Diag diag_   = blas::batch::extract<Diag>(diag, i);
            int64_t m_   = blas::batch::extract<int64_t>(m, i);
            int64_t n_   = blas::batch::extract<int64_t>(n, i);
            int64_t lda_ = blas::batch::extract<int64_t>(ldda, i);
            int64_t ldb_ = blas::batch::extract<int64_t>(lddb, i);
            std::complex<double> alpha_ = blas::batch::extract<std::complex<double> >(alpha, i);
            std::complex<double>* dA_   = blas::batch::extract<std::complex<double>*>(Aarray, i);
            std::complex<double>* dB_   = blas::batch::extract<std::complex<double>*>(Barray, i);
            blas::trsm(
                layout, side_, uplo_, trans_, diag_, m_, n_,
                alpha_, dA_, lda_,
                        dB_, ldb_, queue );
            queue.revolve();
        }
        queue.join();
    }
}
