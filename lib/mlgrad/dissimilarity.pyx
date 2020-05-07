# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=True

# The MIT License (MIT)
#
# Copyright (c) <2015-2019> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from libc.math cimport fabs, pow, sqrt, fmax, exp, log
from libc.math cimport isnan, isinf
from libc.stdlib cimport strtod

cdef double c_nan = strtod("NaN", NULL)
cdef double c_inf = strtod("Inf", NULL)

cdef class Dissimilarity(object):   
    #
    def __init__(self, func):
        self.with_scale = 0
        self.func = func
    #
    cdef double evaluate(self, double z, double u):
        cdef double err = z - u
        return self.func.evaluate(err)
    #
    cdef double derivative_u(self, double z, double u):
        cdef double err = z - u
        return -self.func.derivative(err)
    #
    cdef double derivative_z(self, double z, double u):
        cdef double err = z - u
        return self.func.derivative(err)
    #
    cdef double derivative_uz(self, double z, double u):
        cdef double err = z - u
        return -self.func.derivative2(err)
    #
    cdef double derivative_uu(self, double z, double u):
        cdef double err = z - u
        return self.func.derivative2(err)
