// Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "blas.hh"

namespace blas {

//------------------------------------------------------------------------------
/// @return BLAS++ version.
/// Version is integer of form yyyymmrr, where yyyy is year, mm is month,
/// and rr is release counter within month, starting at 00.
///
int blaspp_version()
{
    return BLASPP_VERSION;
}

// BLASPP_ID is the Mercurial or git commit hash ID, either
// defined by `git rev-parse --short HEAD` in Makefile,
// or defined here by make_release.py for release tar files. DO NOT EDIT.
#ifndef BLASPP_ID
#define BLASPP_ID "unknown"
#endif

//------------------------------------------------------------------------------
/// @return BLAS++ Mercurial or git commit hash ID.
///
const char* blaspp_id()
{
    return BLASPP_ID;
}

}  // namespace blas
