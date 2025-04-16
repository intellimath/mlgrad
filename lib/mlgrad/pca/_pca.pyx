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

cdef double _S_norm(double[:,::1] S, double[::1] a) noexcept nogil:
    cdef Py_ssize_t i, j, k, n = S.shape[0]
    cdef double ai, s
    cdef double *Si
    cdef double *aa = &a[0]

    s = 0
    for i in range(n):
        ai = aa[i]
        Si = &S[i,0]
        for j in range(n):
            s += ai * Si[j] * aa[j]
    return s

cpdef _find_pc(double[:,::1] S, double[::1] a0 = None, 
               Py_ssize_t n_iter=1000, double tol=1.0e-6, bint verbose=0):

    cdef Py_ssize_t i, j, n = S.shape[0]
    cdef Py_ssize_t K = 0
    cdef double[::1] a
    cdef double *aa
    cdef double[::1] S_a = inventory.empty_array(n)
    cdef double *SS_a = &S_a[0]
    cdef double *SS_i
    cdef double na, s, v
    cdef double L, L_prev
    
    if a0 is None:
        a = np.random.random(S.shape[0])
    else:
        a = a0

    aa = &a[0]

    inventory._normalize2(aa, n)

    L = _S_norm(S, a)

    K = 1
    while K < n_iter:
        L_prev = L

        for i in range(n):
            SS_i = &S[i, 0]
            s = 0
            for j in range(n):
                s += SS_i[j] * aa[j]
            SS_a[i] = s

        for i in range(n):
            aa[i] = SS_a[i] / L

        inventory._normalize2(aa, n)

        L = 0
        for i in range(n):
            L += SS_a[i] * aa[i]
        
        if fabs(L - L_prev) / fabs(L) < tol:
            break        

        K += 1

    ra = inventory._asarray(a)      
                
    if verbose:
        print("K:", K, "L:", L, "PC:", ra)
            
    return ra, L

cdef void _sub_outer(double[:,::1] S, double[::1] a, double L) noexcept nogil:
    cdef Py_ssize_t i, j, n = a.shape[0]
    cdef double *S_i
    cdef double *aa = &a[0]
    cdef double v

    for i in range(n):
        S_i = &S[i,0]
        v = L * aa[i]
        for j in range(n):
            S_i[j] -= v * aa[j]

cpdef _find_pc_all(double[:,::1] S, Py_ssize_t m=-1,
                  Py_ssize_t n_iter=1000, double tol=1.0e-6, bint verbose=0):
    cdef Py_ssize_t j, n = S.shape[0]
    cdef double[:,::1] S1 = S.copy()

    cdef object As = inventory.empty_array2(m, n)
    cdef object Ls = inventory.empty_array(m)
    cdef double[:,::1] AA
    cdef double[::1] LL
    cdef double[::1] a
    cdef double Lj

    if m <= 0:
        m = n

    AA = As
    LL = Ls

    for j in range(m):
        a, Lj = _find_pc(S1, None, n_iter, tol, verbose)
        _sub_outer(S1, a, Lj)
        LL[j] = Lj
        inventory._move(&AA[j,0], &a[0], n)

    return As, Ls
