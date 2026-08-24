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

from libc.math cimport isnan, fma, sqrt, fabs, pow
cimport mlgrad.inventory as inventory
from mlgrad.avragg cimport Average

from cython.parallel cimport parallel, prange

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

cdef double _norm2(double *a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v1, v2, v3, v4, s

    s = 0
    i = 0
    while i + 4 < n:
        v1 = a[0]
        v2 = a[1]
        v3 = a[2]
        v4 = a[3]
        s += v1*v1 + v2*v2 + v3*v3 + v4*v4
        i += 4
        a += 4

    while i < n:
        v1 = a[0]
        s += v1*v1
        a += 1
        i += 1

    s = sqrt(s)
    return s

cdef void _scale(double *a, Py_ssize_t n, double s) noexcept nogil:
    cdef Py_ssize_t i

    i = 0
    while i + 4 < n:
        a[0] *= s
        a[1] *= s
        a[2] *= s
        a[3] *= s
        i += 4
        a += 4

    while i < n:
        a[0] *= s
        a += 1
        i += 1

cdef void _normalize2(double *a, Py_ssize_t n) noexcept nogil:
    cdef double s = _norm2(a, n)
    _scale(a, n, 1.0/s)

cdef double _dot(double *a, double *b, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s

    i = 0
    s = 0
    while i + 4 < n:
        s += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
        a += 4
        b += 4
        i += 4

    while i < n:
        s += a[0] * b[0]
        a += 1
        b += 1
        i += 1

    return s

cdef void _matdot(double *C, double *A, double *b, Py_ssize_t N, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double *A_i
    cdef double s, v

    for i in range(N):
    # for i in prange(n, schedule="static", nogil=True):
        A_i = A + i*n
        C[i] = _dot(A_i, b, n)

cdef void _flip_vector(double *aa, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i, max_i
    cdef double v, max_val

    max_val = 0
    max_i = 0
    for i in range(n):
        v = aa[i]
        if v < 0:
            v = -v
        if v > max_val:
            max_val = v
            max_i = i

    if aa[max_i] < 0:
        for i in range(n):
            aa[i] = -aa[i]

cpdef _find_pc(double[:,::1] S, double[::1] a0 = None,
               Py_ssize_t n_iter=1000, double tol=1.0e-4, bint verbose=0):

    cdef Py_ssize_t i, max_i, j, n = S.shape[0]
    cdef Py_ssize_t K = 0
    cdef double[::1] a
    cdef double *aa
    cdef double[::1] S_a = inventory.empty_array(n)
    cdef double *SS_a = &S_a[0]
    cdef double *SS = &S[0,0]
    cdef double na, s, v, max_val
    cdef double L, L_prev

    arr = inventory.empty_array(n)
    if a0 is None:
        arr[:] = np.random.random(n)
    else:
        arr[:] = a0
    a = arr
    aa = &a[0]

    _normalize2(aa, n)

    _matdot(SS_a, SS, aa, n, n)

    L = _dot(SS_a, aa, n)

    for K in range(n_iter):
        L_prev = L

        _matdot(SS_a, SS, aa, n, n)

        for i in range(n):
            aa[i] = SS_a[i]

        _normalize2(aa, n)

        L = _dot(SS_a, aa, n)

        if fabs(L - L_prev) / fabs(L) < tol:
            break

    K += 1

    _flip_vector(aa, n)

    if verbose:
        print("K:", K, "L:", L)

    return arr, L

cpdef _find_pc_all(double[:,::1] S, Py_ssize_t m=-1,
                  Py_ssize_t n_iter=100, double tol=1.0e-4, bint verbose=0):

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


cpdef _find_robust_pc(double[:,::1] X, Average wma,
                      Py_ssize_t n_iter=100, double tol=1.0e-4,
                      bint verbose=0, list qvals=None):

    cdef Py_ssize_t i, j, k, N = X.shape[0], n = X.shape[1]
    cdef Py_ssize_t K = 0
    cdef double[::1] a, a_min
    cdef double s, v
    cdef double L, L_prev

    cdef double[::1] X2 = inventory.empty_array(N)
    cdef double[::1] U  = inventory.empty_array(N)
    cdef double[::1] Z  = inventory.empty_array(N)
    cdef double[::1] W  = inventory.empty_array(N)
    cdef double[::1] WK  = inventory.empty_array(N)

    cdef double pval, pval_min, pval_min_prev, pval_prev
    cdef bint to_finish = 0
    cdef double Q
    cdef int count = 0

    a = np.random.random(n)
    _normalize2(&a[0], n)
    a_min = a.copy()

    # for k in range(N):
    #     X2[k] = _dot(&X[k,0], &X[k,0], n)

    _matdot(&U[0], &X[0,0], &a[0], N, n)
    for k in range(N):
        v = U[k]
        Z[k] = - v * v

    pval = pval_min = wma._evaluate(Z)
    wma._gradient(Z, W)

    pval_min_prev = pval_min * 10
    Q = 1 + fabs(pval_min)

    if qvals is not None:
        qvals.append(pval)

    to_finish = False
    for K in range(n_iter):
        pval_prev = pval

        for k in range(N):
            WK[k] = W[k] * U[k]

        for i in range(n):
            s = 0
            for k in range(N):
                s += WK[k] * X[k,i]
            a[i] = s
        inventory._normalize2(a)

        _matdot(&U[0], &X[0,0], &a[0], N, n)

        for k in range(N):
            v = U[k]
            Z[k] = - v * v

        pval = wma._evaluate(Z)
        wma._gradient(Z, W)

        if qvals is not None:
            qvals.append(pval)

        if fabs(pval - pval_prev) / Q < tol:
            to_finish = True
        if fabs(pval - pval_min) / Q < tol:
            to_finish = True
        elif fabs(pval_min - pval_min_prev) / Q < tol:
            if count >= 3:
                to_finish = True
            else:
                count += 1

        if pval < pval_min:
            pval_min_prev = pval_min
            pval_min = pval
            a_min = a.copy()
            # L_min = L
            Q = 1 + fabs(pval_min)
            count = 0
        elif pval < pval_min_prev:
            pval_min_prev = pval

        if to_finish:
            break

    _flip_vector(&a_min[0], n)

    ra_min = np.asarray(a_min)

    _matdot(&U[0], &X[0,0], &a_min[0], N, n)
    L = _dot(&U[0], &U[0], N)

    if verbose:
        print("K:", K, "L:", L)

    return ra_min, L

cdef double _power_norm(double *x, Py_ssize_t n, double q) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, s

    s = 0
    for i in range(n):
        v = x[i]
        if v < 0:
            v = -v
        s += pow(v, q)
    s = pow(s, 1.0 / q)
    return s

cdef void _normalize_q(double *a, Py_ssize_t n, double q) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s

    s = _power_norm(a, n, q)
    _scale(a, n, 1.0/s)
    # for i in range(n):
    #     a[i] /= s

cpdef _find_pc_l2_lq(double[:,::1] S, double q, double[::1] a0=None,
                    Py_ssize_t n_iter=100, double tol=1.0e-4, double eps=0, bint verbose=0):
    cdef Py_ssize_t i, j, n = S.shape[0]
    cdef double[::1] a
    cdef double[::1] S_a = inventory.empty_array(n)
    cdef double *SS_a = &S_a[0]
    cdef double *SS = &S[0,0]
    cdef double *aa
    cdef double v, L, L_prev
    # cdef bint flag

    arr = inventory.empty_array(n)
    if a0 is None:
        arr[:] = np.random.random(n)
    else:
        arr[:] = a0
    a = arr
    aa = &a[0]
 
    _normalize_q(aa, n, q)
    _matdot(SS_a, SS, aa, n, n)
    L = _dot(SS_a, aa, n)

    for K in range(n_iter):
        L_prev = L

        for i in range(n):
            v = aa[i]
            if v < 0:
                v = -v
            aa[i] = SS_a[i] * pow(v, 2-q)

        _normalize_q(aa, n, q)

        if eps != 0:
            flag = 0
            for i in range(n):
                v = aa[i]
                if v < 0:
                    v = -v
                if 0 < v < eps:
                    aa[i] = 0
                    flag = 1

            if flag:
                pn = inventory._power_norm(aa, n, q)
                if pn != 0:
                    for i in range(n):
                        aa[i] /= pn

        _matdot(SS_a, SS, aa, n, n)
        L = _dot(SS_a, aa, n)

        if fabs(L_prev - L) / (1 + fabs(L)) < tol:
            break

    _flip_vector(aa, n)

    _normalize2(aa, n)
    _matdot(SS_a, SS, aa, n, n)
    L = _dot(SS_a, aa, n)

    K += 1
    if verbose:
        print(f"K: {K} L: {L}")

    return arr, L

cdef void _sign(double *a, double *b, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v
    
    for i in range(n):
        v = b[i]
        if v > 0:
            a[i] = 1
        elif v < 0:
            a[i] = -1
        else:
            a[i] = 0

def find_pc_l1_l2(double[:,::1] X, double[::1] a0=None, Py_ssize_t n_iter=100, double tol=1.0e-6, verbose=0):
    cdef Py_ssize_t i, j, K = 0
    cdef Py_ssize_t N = X.shape[0], n = X.shape[1]
    cdef double[::1] Xa = inventory.empty_array(N)
    cdef double[::1] sign_Xa = inventory.empty_array(N)
    cdef double[::1] Sa = inventory.empty_array(n)
    cdef double[::1] a
    cdef double s, v, L, L_prev, dL = 0

    arr = inventory.empty_array(n)
    if a0 is None:
        arr[:] = np.random.random(n)
    else:
        arr[:] = a0
    a = arr

    _normalize2(&a[0], n)

    _matdot(&Xa[0], &X[0,0], &a[0], N, n)
    _sign(&sign_Xa[0], &Xa[0], N)

    for i in range(n):
        s = 0
        for j in range(N):
            s += sign_Xa[j] * X[j,i]
        Sa[i] = s

    L = _dot(&Sa[0], &a[0], n)

    for K in range(n_iter):
        L_prev = L

        for i in range(n):
            a[i] = Sa[i]
        _normalize2(&a[0], n)

        _matdot(&Xa[0], &X[0,0], &a[0], N, n)
        _sign(&sign_Xa[0], &Xa[0], N)

        for i in range(n):
            s = 0
            for j in range(N):
                s += sign_Xa[j] * X[j,i]
            Sa[i] = s
    
        L = _dot(&Sa[0], &a[0], n)

        dL = fabs(L_prev - L) / (1 + fabs(L))
        if dL < tol:
            break

    _flip_vector(&a[0], n)

    K += 1
    if verbose:
        print(f"K: {K} L: {L} dL: {dL}")

    return arr, L
