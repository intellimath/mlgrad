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

import numpy as np

cdef double S_cov(double[:,::1] S, double[::1] a) noexcept nogil:
    cdef Py_ssize_t i, j, k, n = S.shape[0]
    cdef double ai, s = 0
    cdef double *Si
    cdef double *aa = &a[0]

    for i in range(n):
        ai = aa[i]
        Si = &S[i,0]
        for j in range(n):
            s += ai * Si[j] * aa[j]
    return s

cdef void normalize(double *aa, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double na = 0
    cdef double v

    for i in range(n):
        v = aa[i]
        na += v * v
    na = sqrt(na)
    for i in range(n):
        aa[i] /= na

cpdef _find_pc(double[:,::1] S, double[::1] a0 = None, 
               Py_ssize_t n_iter=1000, double tol=1.0e-6, bint verbose=0):

    cdef Py_ssize_t i, j, n = S.shape[0]
    cdef Py_ssize_t K = 0
    cdef double[::1] a
    cdef double *aa
    cdef double[::1] S_a = inventory.empty_array(n)
    cdef double *SS_a = &S_a[0]
    cdef double *SS
    cdef double na, s, v
    cdef double L, L_prev
    
    if a0 is None:
        a = np.random.random(S.shape[0])
    else:
        a = a0

    aa = &a[0]

    normalize(aa, n)

    L = PyFloat_GetMax() / 10

    K = 1
    while K < n_iter:
        L_prev = L

        SS = &S[0,0]
        for i in range(n):
            s = 0
            for j in range(n):
                s += SS[j] * aa[j]
            SS_a[i] = s
            SS += n

        L = 0
        for i in range(n):
            L += SS_a[i] * aa[i]

        if fabs(L - L_prev) < tol:
            break

        for i in range(n):
            aa[i] = SS_a[i] / L

        normalize(aa, n)

        K += 1
                
    if verbose:
        print("K:", K, "L:", L, "PC:", np.asarray(a))
            
    return np.asarray(a), L
