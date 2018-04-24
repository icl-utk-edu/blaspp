#ifndef BLAS_CONFIG_H
#define BLAS_CONFIG_H

#include <stdlib.h>

#ifndef blas_int
    #if defined(BLAS_ILP64)
        #define blas_int              long long  /* or int64_t */
    #else
        #define blas_int              int
    #endif
#endif

/* f2c, hence MacOS Accelerate, returns double instead of float
 * for sdot, slange, clange, etc. */
#if defined(HAVE_MACOS_ACCELERATE) || defined(HAVE_F2C)
    typedef double blas_float_return;
#else
    typedef float blas_float_return;
#endif

#if defined(__cplusplus) && defined(BLAS_COMPLEX_CPP)

#include <complex>
#define blas_complex_float std::complex<float>
#define blas_complex_double std::complex<double>

#else

/* default is BLAS_COMPLEX_C99 */
#include <complex.h>
#define blas_complex_float    float _Complex
#define blas_complex_double   double _Complex

#endif

#endif /* BLAS_CONFIG_H */
