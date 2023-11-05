// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#ifdef __cplusplus
    #include <iostream>
#else
    #include <stdio.h>
#endif

int main()
{
    // xlc must come before clang
    // clang and icc must come before gcc
    // icpx and icx must come before clang
    const char* compiler =
    #ifdef __cplusplus
        // IBM's documentation says __IBMCPP__,
        // but xlc -qshowmacros shows __ibmxl_version__.
        #if defined(__IBMCPP__) || defined(__ibmxl_version__)
            "xlc++";
        #elif defined(_CRAYC)
            "cray";
        #elif defined(__ICC)
            "icpc";
        #elif defined(__INTEL_LLVM_COMPILER)
            "icpx";
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
        #elif defined(_CRAYC)
            "cray";
        #elif defined(__ICC)
            "icc";
        #elif defined(__INTEL_LLVM_COMPILER)
            "icx";
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
