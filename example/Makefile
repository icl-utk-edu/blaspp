# CXX compiler must match the one used to compiler BLAS++.
# Set it your environment or here.
# Sadly, pkg-config doesn't provide a way to query CXX,
# CXXFLAGS (only cflags), CPPFLAGS, or LDFLAGS.

#CXX = g++

#-------------------------------------------------------------------------------
# Set CXXFLAGS and LIBS
pkg_exists := $(shell pkg-config --exists blas++; echo $$?)
ifeq ($(pkg_exists),0)

  # Get flags from pkg-config.
  CXXFLAGS = $(shell pkg-config --cflags blas++)
  LIBS     = $(shell pkg-config --libs   blas++)
  
else
  $(warning WARNING: pkg-config couldn't find blas++. Using hard-coded flags in Makefile.)

  # BLAS++ not in pkg-config.
  # Here's a hard-coded example using OpenBLAS.
  CXXFLAGS = -I/usr/local/blaspp/include -std=c++11
  LIBS     = -L/usr/local/blaspp/lib$(LIB_SUFFIX) -lblaspp -lopenblas
  
endif

#-------------------------------------------------------------------------------
# Rules
example_gemm: example_gemm.o
	$(CXX) -o $@ $^ $(LIBS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	-rm -f example_gemm *.o

#-------------------------------------------------------------------------------
# Debugging
echo:
	@echo "PKG_CONFIG_PATH $(PKG_CONFIG_PATH)"
	@echo "pkg_exists $(pkg_exists)"
	@echo "CXX        $(CXX)"
	@echo "CXXFLAGS   $(CXXFLAGS)"
	@echo "LIBS       $(LIBS)"
