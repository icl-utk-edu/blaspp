# Copyright (c) 2017-2023, University of Tennessee. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# This program is free software: you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause license. See the accompanying LICENSE file.

# CXX compiler must match the one used to compiler BLAS++.
# Set it in your environment or here.
# Sadly, pkg-config doesn't provide a way to query CXX,
# CXXFLAGS (only cflags), CPPFLAGS, or LDFLAGS.

#-------------------------------------------------------------------------------
# Set CXXFLAGS and LIBS
pkg_exists := $(shell pkg-config --exists blaspp; echo $$?)
ifeq ($(pkg_exists),0)

  # Get flags from pkg-config.
  CXX      = $(shell pkg-config --variable CXX blaspp)
  CXXFLAGS = $(shell pkg-config --cflags blaspp)
  LIBS     = $(shell pkg-config --libs   blaspp)

else
  $(warning WARNING: pkg-config couldn't find blaspp. Using hard-coded flags in Makefile.)

  # BLAS++ not in pkg-config.
  # Here's a hard-coded example using OpenBLAS.
  CXXFLAGS = -I/usr/local/blaspp/include -std=c++11
  LIBS     = -L/usr/local/blaspp/lib$(LIB_SUFFIX) -lblaspp -lopenblas

endif

#-------------------------------------------------------------------------------
# Rules

exe = example_gemm example_util
run = ${addsuffix .run, ${exe}}
txt = ${addsuffix .txt, ${exe}}

.DELETE_ON_ERROR:
.SECONDARY:
.SUFFIXES:
.DEFAULT_GOAL := all
.PRECIOUS: ${txt}

# Serialize everything so that `make test` runs in serial.
.NOTPARALLEL:

all: ${exe}

%: %.o
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	-rm -f ${exe} ${txt} *.o *.d

# CMake uses `make test`, GNU autotools uses `make check`; allow both.
test: check
check: ${run}

# Run example, but don't save results; %.run is dummy filename.
%.run: %
	@echo "----------------------------------------------------------------------"
	./$< ${test_args}

# Run example and save result in .txt file.
txt: ${txt}
%.txt: %
	./$< ${test_args} > $@

#-------------------------------------------------------------------------------
# Debugging
echo:
	@echo "PKG_CONFIG_PATH $(PKG_CONFIG_PATH)"
	@echo "pkg_exists $(pkg_exists)"
	@echo "CXX        $(CXX)"
	@echo "CXXFLAGS   $(CXXFLAGS)"
	@echo "LIBS       $(LIBS)"
	@echo
	@echo "exe       $(exe)"
	@echo "run       $(run)"
	@echo "txt       $(txt)"
