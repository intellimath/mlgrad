# coding: utf-8 

# The MIT License (MIT)
#
# Copyright (c) <2015-2023> <Shibzukhov Zaur, szport at gmail dot com>
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

from libc.math cimport isnan, fma, sqrt, fabs

cimport mlgrad.inventory as inventory
from mlgrad.avragg cimport Average

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()   


cdef double _S_norm(double[:,::1] S, double[::1] a) noexcept nogil


cpdef _find_pc(double[:,::1] S, double[::1] a0=*, 
               Py_ssize_t n_iter=*, double tol=*, bint verbose=*)

cpdef _find_robust_pc(double[:,::1] X, Average wma,  
                      Py_ssize_t n_iter=*, double tol=*, bint verbose=*, list qvals=*, str mode=*)
cpdef _find_robust_pc_min(double[:,::1] X, Average wma,  
                      Py_ssize_t n_iter=*, double tol=*, bint verbose=*, list qvals=*)
cpdef _find_robust_pc_max(double[:,::1] X, Average wma,  
                      Py_ssize_t n_iter=*, double tol=*, bint verbose=*, list qvals=*)


cpdef _find_pc_all(double[:,::1] S, Py_ssize_t m=*,
                  Py_ssize_t n_iter=*, double tol=*, bint verbose=*)
