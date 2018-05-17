#ifndef BATCH_SUPPORT_HH
#define BATCH_SUPPORT_HH

namespace blas
{

namespace batch
{

// -----------------------------------------------------------------------------
// gemm (blaspp side)
template<typename T>
inline
bool is_supported( 
        const char* target, 
        std::vector<blas::Op> const &transA, std::vector<blas::Op> const &transB,
        std::vector<int64_t>  const &m,      std::vector<int64_t> const &n,    std::vector<int64_t> const &k,
        std::vector<T >   const &alpha,
        std::vector<T*>   const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>   const &Barray, std::vector<int64_t> const &ldb,
        std::vector<T >   const &beta,
        std::vector<T*>   const &Carray, std::vector<int64_t> const &ldc, 
        int64_t batch);

// -----------------------------------------------------------------------------
// gemm (vendor side)
template<typename T>
inline
bool is_vendor_supported(
        const char* target, 
        std::vector<blas::Op> const &transA, std::vector<blas::Op> const &transB,
        std::vector<int64_t>  const &m,      std::vector<int64_t> const &n,    std::vector<int64_t> const &k,
        std::vector<T >   const &alpha,
        std::vector<T*>   const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>   const &Barray, std::vector<int64_t> const &ldb,
        std::vector<T >   const &beta,
        std::vector<T*>   const &Carray, std::vector<int64_t> const &ldc, 
        int64_t batch);

// -----------------------------------------------------------------------------
// trsm (blaspp side)
template<typename T>
inline
bool is_supported( 
        const char* target, 
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T >         const &alpha,
        std::vector<T*>         const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>         const &Barray, std::vector<int64_t> const &ldb,
        const size_t batch);

// -----------------------------------------------------------------------------
// trsm (vendor side)
template<typename T>
inline
bool is_vendor_supported( 
        const char* target, 
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T >         const &alpha,
        std::vector<T*>         const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>         const &Barray, std::vector<int64_t> const &ldb,
        const size_t batch);

}    // namespace batch

}    // namespace blas

// -----------------------------------------------------------------------------
// gemm (blaspp side)
template<typename T>
inline
bool blas::batch::is_supported( 
        const char* target, 
        std::vector<blas::Op> const &transA, std::vector<blas::Op> const &transB,
        std::vector<int64_t>  const &m,      std::vector<int64_t> const &n,    std::vector<int64_t> const &k,
        std::vector<T >   const &alpha,
        std::vector<T*>   const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>   const &Barray, std::vector<int64_t> const &ldb,
        std::vector<T >   const &beta,
        std::vector<T*>   const &Carray, std::vector<int64_t> const &ldc, 
        int64_t batch)
{
    blas_error_if( strcmp(target, "device") != 0 );
    bool supported = false; 

    supported |= ( transA.size() == 1     && 
                   transB.size() == 1     && 
                   m.size()      == 1     && 
                   n.size()      == 1     && 
                   k.size()      == 1     &&
                   alpha.size()  == 1     && 
                   Aarray.size() == batch && 
                   lda.size()    == 1     &&
                   Barray.size() == batch && 
                   ldb.size()    == 1     &&
                   beta.size()   == 1     && 
                   Carray.size() == batch && 
                   ldc.size()    == 1
                 );

    return supported;
}

// -----------------------------------------------------------------------------
// gemm (vendor side)
template<typename T>
inline
bool blas::batch::is_vendor_supported(
        const char* target, 
        std::vector<blas::Op> const &transA, std::vector<blas::Op> const &transB,
        std::vector<int64_t>  const &m,      std::vector<int64_t> const &n,    std::vector<int64_t> const &k,
        std::vector<T >   const &alpha,
        std::vector<T*>   const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>   const &Barray, std::vector<int64_t> const &ldb,
        std::vector<T >   const &beta,
        std::vector<T*>   const &Carray, std::vector<int64_t> const &ldc, 
        int64_t batch)
{
    blas_error_if( strcmp(target, "device") != 0 );
    bool supported = false; 

        supported |= ( transA.size() == 1     && 
                   transB.size() == 1     && 
                   m.size()      == 1     && 
                   n.size()      == 1     && 
                   k.size()      == 1     &&
                   alpha.size()  == 1     && 
                   Aarray.size() == batch && 
                   lda.size()    == 1     &&
                   Barray.size() == batch && 
                   ldb.size()    == 1     &&
                   beta.size()   == 1     && 
                   Carray.size() == batch && 
                   ldc.size()    == 1
                 );

    return supported;
}

// -----------------------------------------------------------------------------
// trsm (blaspp side)
template<typename T>
inline
bool blas::batch::is_supported( 
        const char* target, 
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T >         const &alpha,
        std::vector<T*>         const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>         const &Barray, std::vector<int64_t> const &ldb,
        const size_t batch)
{
    blas_error_if( strcmp(target, "device") != 0 );
    bool supported = false; 

    supported |= ( side.size()   == 1     && 
                   uplo.size()   == 1     && 
                   trans.size()  == 1     && 
                   diag.size()   == 1     && 
                   m.size()      == 1     && 
                   n.size()      == 1     && 
                   alpha.size()  == 1     && 
                   Aarray.size() == batch && 
                   lda.size()    == 1     &&
                   Barray.size() == batch && 
                   ldb.size()    == 1
                   );

    return supported;
}

// -----------------------------------------------------------------------------
// trsm (vendor side)
template<typename T>
inline
bool blas::batch::is_vendor_supported( 
        const char* target, 
        std::vector<blas::Side> const &side,
        std::vector<blas::Uplo> const &uplo,
        std::vector<blas::Op>   const &trans,
        std::vector<blas::Diag> const &diag,
        std::vector<int64_t>    const &m, 
        std::vector<int64_t>    const &n, 
        std::vector<T >         const &alpha,
        std::vector<T*>         const &Aarray, std::vector<int64_t> const &lda,
        std::vector<T*>         const &Barray, std::vector<int64_t> const &ldb,
        const size_t batch)
{
    blas_error_if( strcmp(target, "device") != 0 );
    bool supported = false; 

    supported |= ( side.size()   == 1     && 
                   uplo.size()   == 1     && 
                   trans.size()  == 1     && 
                   diag.size()   == 1     && 
                   m.size()      == 1     && 
                   n.size()      == 1     && 
                   alpha.size()  == 1     && 
                   Aarray.size() == batch && 
                   lda.size()    == 1     &&
                   Barray.size() == batch && 
                   ldb.size()    == 1
                   );

    return supported;
}

#endif    // BATCH_SUPPORT_HH