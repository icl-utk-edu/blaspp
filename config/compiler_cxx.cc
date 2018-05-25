#ifdef __cplusplus
    #include <iostream>
#else
    #include <stdio.h>
#endif

int main()
{
    // xlc must come before clang; clang and icc must come before gcc
    const char* compiler =
    #ifdef __cplusplus
        // IBM's documentation says __IBMCPP__,
        // but xlc -qshowmacros shows __ibmxl_version__.
        #if defined(__IBMCPP__) || defined(__ibmxl_version__)
            "xlc++";
        #elif defined(__ICC)
            "icpc";
        #elif defined(_MSC_VER)
            "MSC";
        #elif defined(__clang__)
            "clang++";
        #elif defined(__GNUG__)
            "g++";
        #else
            "unknown C++";
        #endif
    #else
        #if defined(__IBMC__) || defined(__ibmxl_version__)
            "xlc";
        #elif defined(__ICC)
            "icc";
        #elif defined(_MSC_VER)
            "MSC";
        #elif defined(__clang__)
            "clang";
        #elif defined(__GNUC__)
            "gcc";
        #else
            "unknown C";
        #endif
    #endif

    #ifdef __cplusplus
        std::cout << compiler << "\n";
    #else
        printf( "%s\n", compiler );
    #endif
    return 0;
}
