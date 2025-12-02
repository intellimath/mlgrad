# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2025> <Shibzukhov Zaur, szport at gmail dot com>
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

from libc.math cimport isnan, fma, sqrt, fabs
cimport mlgrad.inventory as inventory
from mlgrad.avragg cimport Average

cdef double _S_norm(double[:,::1] S, double[::1] a) noexcept nogil:
    cdef Py_ssize_t i, j, k, n = S.shape[0]
    cdef double a_i, s
    cdef double *S_i
    cdef double *aa = &a[0]

    s = 0
    for i in range(n):
        a_i = aa[i]
        S_i = &S[i,0]
        for j in range(n):
            s += a_i * S_i[j] * aa[j]
    return s

cpdef _find_pc(double[:,::1] S, double[::1] a0 = None,
               Py_ssize_t n_iter=1000, double tol=1.0e-5, bint verbose=0):

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

    inventory._normalize2(a)

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

        inventory._normalize2(a)

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

cpdef _find_robust_pc(double[:,::1] X, Average wma,
                      Py_ssize_t n_iter=1000, double tol=1.0e-5,
                      bint verbose=0, list qvals=None):

    cdef Py_ssize_t i, j, k, N = X.shape[0], n = X.shape[1]
    cdef Py_ssize_t K = 0
    cdef double[::1] a, a_min
    # cdef double *aa
    # cdef double[::1] S_a = inventory.empty_array(n)
    # cdef double *SS_a = &S_a[0]
    # cdef double *SS_i
    cdef double s, v
    cdef double L, L_prev

    cdef double[::1] X2 = inventory.empty_array(N)
    cdef double[::1] U  = inventory.empty_array(N)
    cdef double[::1] Z  = inventory.empty_array(N)
    cdef double[::1] W  = inventory.empty_array(N)

    cdef double pval, pval_min, pval_min_prev, pval_prev
    cdef bint to_finish = 0
    cdef double Q
    cdef int count = 0

    a = np.random.random(n)
    inventory._normalize2(a)
    a_min = a.copy()

    for k in range(N):
        X2[k] = inventory._dot(&X[k,0], &X[k,0], n)

    inventory.matdot(U, X, a)
    for k in range(N):
        v = U[k]
        Z[k] = X2[k] - v * v

    pval = pval_min = wma._evaluate(Z)
    wma._gradient(Z, W)

    pval_min_prev = pval_min * 10
    Q = 1 + fabs(pval_min)

    if qvals is not None:
        qvals.append(pval)

    L = 0
    for k in range(N):
        v = U[k]
        L += W[k] * v * v
    L_min = L

    to_finish = False
    for K in range(n_iter):
        pval_prev = pval

        for i in range(n):
            s = 0
            for k in range(N):
                s += W[k] * U[k] * X[k,i]
            a[i] = s
        inventory._normalize2(a)

        inventory.matdot(U, X, a)

        for k in range(N):
            v = U[k]
            Z[k] = X2[k] - v * v

        pval = wma._evaluate(Z)
        wma._gradient(Z, W)

        if qvals is not None:
            qvals.append(pval)

        L = 0
        for k in range(N):
            v = U[k]
            L += W[k] * v * v

        if fabs(pval - pval_prev) / Q < tol:
            to_finish = True
        if fabs(pval - pval_min) / Q < tol:
            to_finish = True
        elif fabs(pval - pval_min_prev) / Q < tol:
            if count >= 3:
                to_finish = True
            else:
                count += 1

        if pval < pval_min:
            pval_min_prev = pval_min
            pval_min = pval
            a_min = a.copy()
            L_min = L
            Q = 1 + fabs(pval_min)
            count = 0
        elif pval < pval_min_prev:
            pval_min_prev = pval

        if to_finish:
            break

    ra_min = np.asarray(a_min)

    if verbose:
        print("K:", K, "L:", L, "PC:", ra_min)

    return ra_min, L_min

cpdef _find_pc_all(double[:,::1] S, Py_ssize_t m=-1,
                  Py_ssize_t n_iter=1000, double tol=1.0e-6, bint verbose=0):
    cdef Py_ssize_t i, j, n = S.shape[0]

    cdef object As = inventory.empty_array2(m, n)
    cdef object Ls = inventory.empty_array(m)
    cdef double[:,::1] AA = As
    cdef double[::1] LL = Ls
    cdef double[::1] a
    cdef double v, L_j

    if m <= 0:
        m = n

    for j in range(m):
        a, L_j = _find_pc(S, a0=None, n_iter=n_iter, tol=tol, verbose=verbose)

        LL[j] = L_j
        inventory._move(&AA[j,0], &a[0], n)

        for i in range(n):
            v = a[i]
            S[i,i] -= L_j * v * v
        for i in range(n-1):
            for j in range(i+1,n):
                v = L_j * a[i] * a[j]
                S[i,j] -= v
                S[j,i] -= v

    return As, Ls

