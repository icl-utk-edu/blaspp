#ifndef BLAS_MANGLING_HH
#define BLAS_MANGLING_HH

// -----------------------------------------------------------------------------
// Fortran name mangling depends on compiler.
// Define FORTRAN_UPPER for uppercase,
// define FORTRAN_LOWER for lowercase (IBM xlf),
// else the default is lowercase with appended underscore
// (GNU gcc, Intel icc, PGI pgfortan, Cray ftn).
#ifndef BLAS_FORTRAN_NAME
    #if defined(FORTRAN_UPPER)
        #define BLAS_FORTRAN_NAME( lower, UPPER ) UPPER
    #elif defined(FORTRAN_LOWER)
        #define BLAS_FORTRAN_NAME( lower, UPPER ) lower
    #else
        #define BLAS_FORTRAN_NAME( lower, UPPER ) lower ## _
    #endif
#endif

#endif        //  #ifndef BLAS_MANGLING_HH
