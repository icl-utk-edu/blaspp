# Usage:
# make by default:
#    - Runs configure.py to create make.inc, if it doesn't exist.
#    - Compiles lib/libblaspp.so, or lib/libblaspp.a (if static=1).
#    - Compiles the tester, test/tester.
#
# make config    - Runs configure.py to create make.inc.
# make lib       - Compiles lib/libblaspp.so, or libblaspp.a (if static=1).
# make tester      - Compiles the tester, test/tester.
# make docs      - Compiles Doxygen documentation.
# make install   - Installs the library and headers to $prefix.
# make clean     - Deletes all objects, libraries, and the tester.
# make distclean - Also deletes make.inc and dependency files (*.d).

#-------------------------------------------------------------------------------
# Configuration
# Variables defined in make.inc, or use make's defaults:
#   CXX, CXXFLAGS   -- C compiler and flags
#   LDFLAGS, LIBS   -- Linker options, library paths, and libraries
#   AR, RANLIB      -- Archiver, ranlib updates library TOC
#   prefix          -- where to install BLAS++

ifeq ($(MAKECMDGOALS),config)
    # For `make config`, don't include make.inc with previous config;
    # force re-creating make.inc.
    .PHONY: config
    config: make.inc

    make.inc: force

    force: ;
else ifneq ($(findstring clean,$(MAKECMDGOALS)),clean)
    # For `make clean` or `make distclean`, don't include make.inc,
    # which could generate it. Otherwise, include make.inc.
    include make.inc
endif

make.inc:
	python configure.py

# Defaults if not given in make.inc. GNU make doesn't have defaults for these.
RANLIB   ?= ranlib
prefix   ?= /usr/local/blaspp

# auto-detect OS
# $OSTYPE may not be exported from the shell, so echo it
ostype = $(shell echo $${OSTYPE})
ifneq ($(findstring darwin, $(ostype)),)
    # MacOS is darwin
    macos = 1
endif

#-------------------------------------------------------------------------------
# if shared
ifneq ($(static),1)
    CXXFLAGS += -fPIC
    LDFLAGS  += -fPIC
    lib_ext = so
else
    lib_ext = a
endif

#-------------------------------------------------------------------------------
# MacOS needs shared library's path set
ifeq ($(macos),1)
    install_name = -install_name @rpath/$(notdir $@)
else
    install_name =
endif

#-------------------------------------------------------------------------------
# Files

ifeq ($(devtarget),cuda)
    lib_src  = $(wildcard src/*.cc)
else
    lib_src  = $(filter-out src/device_%.cc, $(wildcard src/*.cc))
endif
lib_obj  = $(addsuffix .o, $(basename $(lib_src)))
dep     += $(addsuffix .d, $(basename $(lib_src)))

ifeq ($(devtarget),cuda)
    tester_src = $(wildcard test/*.cc)
else
    tester_src = $(filter-out test/%_device.cc, $(wildcard test/*.cc))
endif
tester_obj = $(addsuffix .o, $(basename $(tester_src)))
dep       += $(addsuffix .d, $(basename $(tester_src)))

tester     = test/tester

testsweeper_dir = $(wildcard ../testsweeper)
ifeq ($(testsweeper_dir),)
    testsweeper_dir = $(wildcard ./testsweeper)
endif
ifeq ($(testsweeper_dir),)
    $(tester_obj):
		$(error Tester requires TestSweeper, which was not found. Run 'make config' \
		        or download manually from https://bitbucket.org/icl/testsweeper/)
endif

testsweeper_src = $(wildcard $(testsweeper_dir)/testsweeper.cc $(testsweeper_dir)/testsweeper.hh)

testsweeper = $(testsweeper_dir)/libtestsweeper.$(lib_ext)

#-------------------------------------------------------------------------------
# BLAS++ specific flags and libraries
CXXFLAGS += -I./include

# additional flags and libraries for testers
$(tester_obj): CXXFLAGS += -I$(testsweeper_dir)

TEST_LDFLAGS += -L./lib -Wl,-rpath,$(abspath ./lib)
TEST_LDFLAGS += -L$(testsweeper_dir) -Wl,-rpath,$(abspath $(testsweeper_dir))
TEST_LIBS    += -lblaspp -ltestsweeper

#-------------------------------------------------------------------------------
# Rules
.DELETE_ON_ERROR:
.SUFFIXES:
.PHONY: all lib src test tester headers include docs clean distclean
.DEFAULT_GOAL = all

all: lib tester

install: lib
	mkdir -p $(DESTDIR)$(prefix)/include
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	cp include/*.{h,hh} $(DESTDIR)$(prefix)/include
	cp $(lib) $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)

uninstall:
	$(RM) $(addprefix $(DESTDIR)$(prefix)/, $(headers))
	$(RM) $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/libblaspp.*

#-------------------------------------------------------------------------------
# if re-configured, recompile everything
$(lib_obj) $(tester_obj): make.inc

#-------------------------------------------------------------------------------
# BLAS++ library
lib_a  = lib/libblaspp.a
lib_so = lib/libblaspp.so
lib    = lib/libblaspp.$(lib_ext)

$(lib_so): $(lib_obj)
	mkdir -p lib
	$(CXX) $(LDFLAGS) -shared $(install_name) $(lib_obj) $(LIBS) -o $@

$(lib_a): $(lib_obj)
	mkdir -p lib
	$(RM) $@
	$(AR) cr $@ $(lib_obj)
	$(RANLIB) $@

# sub-directory rules
lib src: $(lib)

lib/clean src/clean:
	$(RM) lib/*.{a,so} src/*.o

#-------------------------------------------------------------------------------
# TestSweeper library
ifneq ($(testsweeper_dir),)
    $(testsweeper): $(testsweeper_src)
		cd $(testsweeper_dir) && $(MAKE) lib CXX=$(CXX)
endif

#-------------------------------------------------------------------------------
# tester
$(tester): $(tester_obj) $(lib) $(testsweeper)
	$(CXX) $(TEST_LDFLAGS) $(LDFLAGS) $(tester_obj) \
		$(TEST_LIBS) $(LIBS) -o $@

# sub-directory rules
# Note 'test' is sub-directory rule; 'tester' is CMake-compatible rule.
test: $(tester)
tester: $(tester)

test/clean:
	$(RM) $(tester) test/*.o

#-------------------------------------------------------------------------------
# headers
# precompile headers to verify self-sufficiency
headers     = $(wildcard include/*.h include/*.hh test/*.hh)
headers_gch = $(addsuffix .gch, $(basename $(headers)))

headers: $(headers_gch)

headers/clean:
	$(RM) $(headers_gch)

# sub-directory rules
include: headers

include/clean: headers/clean

#-------------------------------------------------------------------------------
# documentation
docs:
	doxygen docs/doxygen/doxyfile.conf
	@echo ========================================
	cat docs/doxygen/errors.txt
	@echo ========================================
	@echo "Documentation available in docs/html/index.html"
	@echo ========================================

# sub-directory redirects
src/docs: docs
test/docs: docs

#-------------------------------------------------------------------------------
# general rules
clean: lib/clean test/clean headers/clean

distclean: clean
	$(RM) make.inc $(dep)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $< -o $@

# preprocess source
%.i: %.cc
	$(CXX) $(CXXFLAGS) -I$(testsweeper_dir) -E $< -o $@

# precompile header to check for errors
%.gch: %.h
	$(CXX) $(CXXFLAGS) -I$(testsweeper_dir) -c $< -o $@

%.gch: %.hh
	$(CXX) $(CXXFLAGS) -I$(testsweeper_dir) -c $< -o $@

-include $(dep)

#-------------------------------------------------------------------------------
# debugging
echo:
	@echo "static        = '$(static)'"
	@echo
	@echo "lib_a         = $(lib_a)"
	@echo "lib_so        = $(lib_so)"
	@echo "lib           = $(lib)"
	@echo
	@echo "lib_src       = $(lib_src)"
	@echo
	@echo "lib_obj       = $(lib_obj)"
	@echo
	@echo "tester_src    = $(tester_src)"
	@echo
	@echo "tester_obj    = $(tester_obj)"
	@echo
	@echo "tester        = $(tester)"
	@echo
	@echo "dep           = $(dep)"
	@echo
	@echo "testsweeper_dir   = $(testsweeper_dir)"
	@echo "testsweeper_src   = $(testsweeper_src)"
	@echo "testsweeper       = $(testsweeper)"
	@echo
	@echo "CXX           = $(CXX)"
	@echo "CXXFLAGS      = $(CXXFLAGS)"
	@echo
	@echo "LDFLAGS       = $(LDFLAGS)"
	@echo "LIBS          = $(LIBS)"
	@echo
	@echo "TEST_LDFLAGS  = $(TEST_LDFLAGS)"
	@echo "TEST_LIBS     = $(TEST_LIBS)"
