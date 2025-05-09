// Copyright (c) 2017-2024, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include <atomic>
#include <cstdint>

int main( int argc, char** argv )
{
    std::atomic<std::int64_t> x = 0;
    for (int i = 1; i < argc; ++i) {
        ++x;
    }
    return x;
}
