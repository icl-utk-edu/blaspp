#!/usr/bin/perl -p
#
# Usage: ./errors.pl errors.txt > errors2.txt
#
# Removes extraneous errors from Doxygen log.

s/.* warning: Member \w+\(.*(float|double|std::complex).*\) \(function\) of namespace blas is not documented\.\n//;
s/.* warning: Member (real_t|scalar_t) \(typedef\) of class blas::traits\d* is not documented\.\n//;

#s/.* warning: Member \w+ \(variable\) of class Params is not documented\.\n//;
#s/.* warning: Compound (blas::traits.*|blas::Error|Params) is not documented\.\n//;
