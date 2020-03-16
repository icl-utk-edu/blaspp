     _ ) |      \    __|    |     |
     _ \ |     _ \ \__ \ __ __|__ __|
    ___/____|_/  _\____/   _|    _|

**C++ API for the Basic Linear Algebra Subroutines**

**Innovative Computing Laboratory**

**University of Tennessee**

[TOC]

* * *

About
--------------------------------------------------------------------------------

The Basic Linear Algebra Subprograms (BLAS) have been around for many decades
and serve as the _de facto_ standard for performance-portable and numerically
robust implementation of essential linear algebra functionality.
Originally, they were written in Fortran, and later furnished with a C API
(CBLAS).

The objective of BLAS++ is to provide a convenient, performance oriented API
for development in the C++ language, that, for the most part,
preserves established conventions, while, at the same time, takes advantages
of modern C++ features, such as: namespaces, templates, exceptions, etc.

BLAS++ is part of the SLATE project
([Software for Linear Algebra Targeting Exascale](http://icl.utk.edu/slate/)),
which is funded by the [Department of Energy](https://energy.gov)
as part of its [Exascale Computing Initiative](https://exascaleproject.org)
(ECP).
Closely related to BLAS++ is the
[LAPACK++](https://bitbucket.org/icl/lapackpp) project,
which provides a C++ API for LAPACK.

![BLASPP](http://icl.bitbucket.io/slate/artwork/Bitbucket/blaspp_stack.png)

* * *

Documentation
--------------------------------------------------------------------------------

* [INSTALL.md](INSTALL.md) for installation notes.
* [BLAS++ Doxygen](https://icl.bitbucket.io/blaspp/doxygen/html/)
* [SLATE Working Note 2: C++ API for BLAS and LAPACK](http://www.icl.utk.edu/publications/swan-002)
* [SLATE Working Note 4: C++ API for Batch BLAS](http://www.icl.utk.edu/publications/swan-004)

* * *

Getting Help
--------------------------------------------------------------------------------

For assistance with SLATE, email *slate-user@icl.utk.edu*.
You can also join the *SLATE User* Google group by going to
https://groups.google.com/a/icl.utk.edu/forum/#!forum/slate-user
signing in with your Google credentials, and then clicking `Join group`.

* * *

Resources
--------------------------------------------------------------------------------

* Visit the [LAPACK++ repository](https://bitbucket.org/icl/lapackpp)
  for more information about the C++ API for LAPACK.
* Visit the [SLATE website](http://icl.utk.edu/slate/)
  for more information about the SLATE project.
* Visit the [SLATE Working Notes](http://www.icl.utk.edu/publications/series/swans)
  to find out more about ongoing SLATE developments.
* Visit the [ECP website](https://exascaleproject.org)
  to find out more about the DOE Exascale Computing Initiative.

* * *

Contributing
--------------------------------------------------------------------------------

The SLATE project welcomes contributions from new developers.
Contributions can be offered through the standard Bitbucket pull request model.
We strongly encourage you to coordinate large contributions with the SLATE
development team early in the process.

* * *

Acknowledgments
--------------------------------------------------------------------------------

<!--
https://www.exascaleproject.org/resources/
https://www.olcf.ornl.gov/olcf-media/media-assets/
https://www.alcf.anl.gov/support-center/facility-policies/alcf-acknowledgement-policy
-->

This research was supported by the Exascale Computing Project (17-SC-20-SC), a
joint project of the U.S. Department of Energy's Office of Science and National
Nuclear Security Administration, responsible for delivering a capable exascale
ecosystem, including software, applications, and hardware technology, to support
the nation’s exascale computing imperative.

This research uses resources of the Oak Ridge Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC05-00OR22725.
This research also uses resources of the Argonne Leadership Computing Facility,
which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.

* * *

License
--------------------------------------------------------------------------------

Copyright (c) 2017-2020, University of Tennessee. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

* Neither the name of the University of Tennessee nor the
  names of its contributors may be used to endorse or promote products
  derived from this software without specific prior written permission.

**This software is provided by the copyright holders and contributors "as is" and
any express or implied warranties, including, but not limited to, the implied
warranties of merchantability and fitness for a particular purpose are
disclaimed. In no event shall the copyright holders or contributors be liable
for any direct, indirect, incidental, special, exemplary, or consequential
damages (including, but not limited to, procurement of substitute goods or
services; loss of use, data, or profits; or business interruption) however
caused and on any theory of liability, whether in contract, strict liability, or
tort (including negligence or otherwise) arising in any way out of the use of
this software, even if advised of the possibility of such damage.**
