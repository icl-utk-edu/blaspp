// Copyright (c) 2017-2021, University of Tennessee. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// This program is free software: you can redistribute it and/or modify it under
// the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

#include "testsweeper.hh"

namespace testsweeper {

//------------------------------------------------------------------------------
/// @return TestSweeper version.
/// Version is integer of form yyyymmrr, where yyyy is year, mm is month,
/// and rr is release counter within month, starting at 00.
///
int version()
{
    return TESTSWEEPER_VERSION;
}

// TESTSWEEPER_ID is the Mercurial or git commit hash ID, either
// defined by `hg id` or `git rev-parse --short HEAD` in Makefile,
// or defined here by make_release.py for release tar files. DO NOT EDIT.
#ifndef TESTSWEEPER_ID
#define TESTSWEEPER_ID "unknown"
#endif

//------------------------------------------------------------------------------
/// @return TestSweeper Mercurial or git commit hash ID.
///
const char* id()
{
    return TESTSWEEPER_ID;
}

}  // namespace testsweeper
