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

from cython cimport view
from openmp cimport omp_get_num_procs, omp_get_num_threads
from cpython.object cimport PyObject

from sys import float_info as _float_info

cimport numpy
numpy.import_array()

import numpy as np

from cython.parallel cimport parallel, prange

double_max = _float_info.max
double_min = _float_info.min

cdef public double _double_max = double_max
cdef public double _double_min = _double_min

cdef int num_procs = omp_get_num_procs()
cdef int num_threads = omp_get_num_threads()

cdef int get_num_threads() noexcept nogil:
    return num_threads

cdef int get_num_procs() noexcept nogil:
    return num_procs

cdef int get_num_threads_ex(int m) noexcept nogil:
    if m <= 0:
        m = 1
    if m < num_threads:
        return m
    return num_threads

cdef int get_num_procs_ex(int m) noexcept nogil:
    if m <= 0:
        m = 1
    if m < num_procs:
        return m
    return num_procs

# cdef void set_num_threads(int num) noexcept nogil:
#     num_threads = num

cdef bint _iscontiguousarray(object ob):
    return numpy.PyArray_IS_C_CONTIGUOUS(ob)

cdef bint _isnumpyarray(object ob):
    return numpy.PyArray_CheckExact(ob)

cdef object _asarray(object ob):
    cdef int tp

    if not numpy.PyArray_CheckExact(ob):
        ob = np.array(ob, "d")

    tp = numpy.PyArray_TYPE(ob)
    if tp != numpy.NPY_DOUBLE:
        ob = numpy.PyArray_Cast(<numpy.ndarray>ob, numpy.NPY_DOUBLE)

    if not numpy.PyArray_IS_C_CONTIGUOUS(ob):
        ob = np.ascontiguousarray(ob)

    return ob

def asarray(ob):
    return _asarray(ob)


cdef void init_rand() noexcept nogil:
    srand(time(NULL))    

cdef long rand(long N) noexcept nogil:
    return stdlib_rand() % N

cdef double _min(double *a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i = 1
    cdef double v, a_min = a[0]

    while i < n:
        v = a[i]
        if v < a_min:
            a_min = v
        i += 1

    return a_min

cdef int hasnan(double[::1] a) noexcept nogil:
    return _hasnan(&a[0], a.shape[0])
        
cdef int _hasnan(double *a, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        if isnan(a[i]):
            return 1
    return 0

cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = 0
        
cdef void clear(double[::1] to) noexcept nogil:
    _clear(&to[0], <const Py_ssize_t>to.shape[0])

cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n*m):
        to[i] = 0
        
cdef void clear2(double[:,::1] to) noexcept nogil:
    _clear2(&to[0,0], <const Py_ssize_t>to.shape[0], <const Py_ssize_t>to.shape[1])
    
cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = c

cdef void fill(double[::1] to, const double c) noexcept nogil:
    _fill(&to[0], c, <const Py_ssize_t>to.shape[0])

cdef void _move_t(double *to, const double *src, const Py_ssize_t n, const Py_ssize_t step) noexcept nogil:
    cdef Py_ssize_t i, j

    j = 0
    for i in range(n):
        to[i] = src[j]
        j += step
    
cdef void _move(double *to, const double *src, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = src[i]

cdef void move(double[::1] to, double[::1] src) noexcept nogil:
    _move(&to[0], &src[0], to.shape[0])
    
cdef void move2(double[:, ::1] to, double[:,::1] src) noexcept nogil:
    _move(&to[0,0], &src[0,0], to.shape[0] * to.shape[1])

cdef void move3(double[:,:,::1] to, double[:,:,::1] src) noexcept nogil:
    _move(&to[0,0,0], &src[0,0,0], to.shape[0] * to.shape[1] * to.shape[2])
    
cdef void _add(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        c[i] = a[i] + b[i]

cdef void add(double[::1] c, double[::1] a, double[::1] b) noexcept nogil:
    _add(&c[0], &a[0], &b[0], a.shape[0])
        

cdef void _iadd(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] += b[i]

cdef void iadd(double[::1] a, double[::1] b) noexcept nogil:
    _iadd(&a[0], &b[0], a.shape[0])

cdef void iadd2(double[:,::1] a, double[:,::1] b) noexcept nogil:
    _iadd(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _isub(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] -= b[i]

cdef void isub(double[::1] a, double[::1] b) noexcept nogil:
    _isub(&a[0], &b[0], a.shape[0])

cdef void _sub(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        c[i] = a[i] - b[i]
    
cdef void sub(double[::1] c, double[::1] a, double[::1] b) noexcept nogil:
    _sub(&c[0], &a[0], &b[0], a.shape[0])
    
cdef void _isub_mask(double *a, const double *b, uint8 *m, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        if m[i] == 0:
            a[i] -= b[i]

cdef void isub_mask(double[::1] a, double[::1] b, uint8[::1] m) noexcept nogil:
    _isub_mask(&a[0], &b[0], &m[0], a.shape[0])
    
cdef double _sum(const double *a, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s

cdef double sum(double[::1] a) noexcept nogil:
    return _sum(&a[0], a.shape[0])

cdef void _mul_const(double *a, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] *= c

cdef void mul_const(double[::1] a, const double c) noexcept nogil:
    _mul_const(&a[0], c, a.shape[0])

cdef void mul_const2(double[:,::1] a, const double c) noexcept nogil:
    _mul_const(&a[0,0], c, a.shape[0] * a.shape[1])

cdef void mul_const3(double[:,:,::1] a, const double c) noexcept nogil:
    _mul_const(&a[0,0,0], c, a.shape[0] * a.shape[1] * a.shape[2])
    
cdef void _mul_add(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        a[i] += c * b[i]

cdef void mul_add(double[::1] a, double[::1] b, const double c) noexcept nogil:
    _mul_add(&a[0], &b[0], c, a.shape[0])

cdef void mul_add2(double[:,::1] a, double[:,::1] b, const double c) noexcept nogil:
    _mul_add(&a[0,0], &b[0,0], c, a.shape[0] * a.shape[1])
    
cdef void _mul_set(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        a[i] = c * b[i]

cdef void mul_set(double[::1] a, double[::1] b, double c) noexcept nogil:
    _mul_set(&a[0], &b[0], c, a.shape[0])

cdef void _mul_set1(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    a[0] = c
    a += 1
    for i in range(n):
        a[i] = c * b[i]

cdef void mul_set1(double[::1] a, double[::1] b, double c) noexcept nogil:
    _mul_set(&a[0], &b[0], c, a.shape[0])
    
cdef void _imul(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        a[i] *= b[i]

cdef void imul(double[::1] a, double[::1] b) noexcept nogil:
    _imul(&a[0], &b[0], a.shape[0])

cdef void imul2(double[:,::1] a, double[:,::1] b) noexcept nogil:
    _imul(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _mul(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        c[i] = a[i] * b[i]

cdef void mul(double[::1] c, double[::1] a, double[::1] b) noexcept nogil:
    _mul(&c[0], &a[0], &b[0], a.shape[0])


cdef double dot1(double[::1] a, double[::1] x) noexcept nogil:
    return _dot1(&a[0], &x[0], a.shape[0])

cdef double _dot1(const double *a, const double *x, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = a[0]

    a += 1
    for i in range(n):
        s += a[i] * x[i]
    return s

cdef double dot(double[::1] a, double[::1] x) noexcept nogil:
    return _dot(&a[0], &x[0], a.shape[0])

cdef double _dot(const double *a, const double *x, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * x[i]
    return s

cdef double dot_t(double[::1] a, double[:,::1] b) noexcept nogil:
    return _dot_t(&a[0], &b[0,0], a.shape[0], b.shape[0])

cdef double _dot_t(const double *a, double *b, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[0]
        b += m
    return s

cdef void _matdot(double *output, double *M, const double *X, 
                    const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double *M_j = M
    cdef double s

    # for j in prange(n_output, schedule='static', nogil=True, num_threads=num_procs):
    for j in range(n_output):
        s = 0
        for i in range(n_input):
            s += M_j[i] * X[i]
        output[j] = s
        M_j += n_input

cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil:
    _matdot(&output[0], &M[0,0], &X[0], X.shape[0], output.shape[0])
        
cdef void _matdot2(double *output, double *M, const double *X, 
                   const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double s
    cdef double *M_j = M

    # for j in prange(n_output, schedule='static', nogil=True, num_threads=num_procs):
    for j in range(n_output):
        s = M_j[0]
        M_j += 1
        for i in range(n_input):
            s += M_j[i] * X[i]
        output[j] = s
        M_j += n_input

cdef void matdot2(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil:
    _matdot2(&output[0], &M[0,0], &X[0], <const Py_ssize_t>X.shape[0], <const Py_ssize_t>M.shape[0])
        
cdef void _mul_add_arrays(double *a, double *M, const double *ss, 
                          const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double *Mj = M;
    cdef double sx

    for j in range(n_output):
        Mj += 1
        sx = ss[j]
        for i in range(n_input):
            a[i] += sx * Mj[i]
        Mj += n_input

cdef void mul_add_arrays(double[::1] a, double[:,::1] M, double[::1] ss) noexcept nogil:
    _mul_add_arrays(&a[0], &M[0,0], &ss[0], <const Py_ssize_t>(a.shape[0]), <const Py_ssize_t>(M.shape[0]))
        
cdef void _mul_grad(double *grad, const double *X, const double *ss, 
                    const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double *G = grad
    cdef double sx
    
    for j in range(n_output):
        sx = ss[j]
        G[0] = sx
        G += 1
        for i in range(n_input):
            G[i] = sx * X[i]
        G += n_input+1

cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) noexcept nogil:
    _mul_grad(&grad[0,0], &X[0], &ss[0], <const Py_ssize_t>X.shape[0], <const Py_ssize_t>grad.shape[0])

cdef void _normalize(double *a, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double S

    S = 0
    for i in range(n):
        S += fabs(a[i])

    for i in range(n):
        a[i] /= S

cdef void normalize(double[::1] a) noexcept nogil:
    _normalize(&a[0], a.shape[0])
        
cdef void _normalize2(double *a, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, S

    S = 0
    for i in range(n):
        v = a[i]
        S += v * v

    S = sqrt(S)

    for i in range(n):
        a[i] /= S
        
cdef void normalize2(double[::1] a) noexcept nogil:
    _normalize2(&a[0], a.shape[0])

cdef double _mahalanobis_norm_one(double *S, const double *x, 
                                  const Py_ssize_t n) noexcept nogil:
    cdef double x1, x2
    cdef Py_ssize_t i, j
    cdef double s, x_i, s_i
    cdef double *S_i
    
    if n == 2:
        x1 = x[0]
        x2 = x[1]
        return S[0] * x1 * x1 + \
               S[3] * x2 * x2 + \
               2 * (S[1] * x1 * x2)
    
    s = 0
    S_i = S
    for i in range(n):
        x_i = x[i]           
        s += S_i[i] * x_i * x_i

        s_i = 0
        for j in range(i+1, n):
            s_i += S_i[j] * x[j]

        s += 2 * x_i * s_i

        S_i += n

    return s

cdef _mahalanobis_norm(double[:,::1] S, double[:,::1] X, double[::1] Y):
    cdef Py_ssize_t n = S.shape[0]
    cdef Py_ssize_t k, N = X.shape[0]

    # if X.shape[1] != n:
    #     raise TypeError("X.shape[1] != S.shape[0]")
    # if Y.shape[0] != N:
    #     raise TypeError("Y.shape[1] != X.shape[0]")

    for k in range(N):
        Y[k] = _mahalanobis_norm_one(&S[0,0], &X[k,0], n)

def mahalanobis_norm(S, X):
    Y = empty_array(X.shape[0])
    _mahalanobis_norm(S, X, Y)
    return Y

cdef _mahalanobis_distance(double[:,::1] S, double[:,::1] X, double[::1] c, double[::1] Y):
    cdef Py_ssize_t n = S.shape[0]
    cdef Py_ssize_t k, N = X.shape[0]
    cdef double[::1] v = empty_array(n)

    # if X.shape[1] != n:
    #     raise TypeError("X.shape[1] != S.shape[0]")
    # if Y.shape[0] != N:
    #     raise TypeError("Y.shape[1] != X.shape[0]")
    # if c.shape[0] != n:
    #     raise TypeError("c.shape[0] != S.shape[0]")

    for k in range(N):
        _sub(&v[0], &X[k,0], &c[0], n)
        Y[k] = _mahalanobis_norm_one(&S[0,0], &v[0], n)

def mahalanobis_distance(S, X, c):
    Y = empty_array(X.shape[0])
    _mahalanobis_distance(S, X, c, Y)
    return Y
        
        
cdef void _scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil:
    """
    Вычисление взвешенной ковариационной матрицы
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
    Результат:
       S: матрица (n,n):
          S = (W[0] * outer(X[0,:],X[0,:]) + ... + W[N-1] * outer(X[N-1,:],X[N-1,:]))
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, j, k
        double s, xkj
        double *Xk
        double *ss

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            Xk = &X[0,0]
            s = 0
            for k in range(N):
                s += W[k] * Xk[i] * Xk[j]
                Xk += n
            ss[j] = s
        ss += n

cdef void _scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil:
    """
    Вычисление ковариационной матрицы
    Вход:
       X: матрица (N,n)
    Результат:
       S: матрица (n,n):
          S = (1/N) X.T @ X
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, j, k
        double s
        double *Xk
        double *ss

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            Xk = &X[0,0]
            s = 0
            for k in range(N):
                s += Xk[i] * Xk[j]
                Xk += n
            ss[j] = s
        ss += n

    ss = &S[0,0]
    for i in range(n):
        for j in range(n):
            ss[j] /= N
        ss += n

def scatter_matrix(double[:,::1] X):
    cdef Py_ssize_t n = X.shape[1]
    cdef object S = zeros_array2(n, n)
    _scatter_matrix(X, S)
    return S

def scatter_matrix_weighted(double[:,::1] X, double[::1] W):
    cdef Py_ssize_t n = X.shape[1]
    cdef object S = zeros_array2(n, n)
    _scatter_matrix_weighted(X, W, S)
    return S

# cdef object _outer_1d(double[::1] a, double[::1] b, double c):
#     cdef Py_ssize_t i, j, n = a.shape[0]
    
#     cdef object ret = emmty_array2(n, n)
#     cdef double[:,::1] R = ret
#     cdef double *Ri

#     cdef double *aa = &a[0]
#     cdef double *bb = &b[0]
#     cdef double v

#     for i in range(n):
#         Ri = &R[i,0]
#         v = aa[i]
#         for j in range(n):
#             Ri[j] = c * v * bb[j]

#     return ret

# def outer_1d(a, b, c):
#     return _outer_1d(a, b, c)

cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) noexcept nogil:
    """
    Взвешенная сумма строк матрицы:
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
       Y: массив (N) - результат:
          Y[i] = W[0] * X[0,:] + ... + W[N-1] * X[N-1,:]
    
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, k
        double *Xk
        double *yy = &Y[0]
        double wk, y
    
    for i in range(n):
        y = 0
        Xk = &X[0,i]
        for k in range(N):
            wk = W[k]
            y += wk * Xk[0]
            Xk += n
        yy[i] = y

cdef object empty_array(Py_ssize_t size):
    cdef numpy.npy_intp n = size
    return numpy.PyArray_EMPTY(1, &n, numpy.NPY_DOUBLE, 0)

cdef object filled_array(Py_ssize_t size, double val):
    cdef numpy.npy_intp n = size
    cdef object ob = numpy.PyArray_EMPTY(1, &n, numpy.NPY_DOUBLE, 0)
    cdef double *data = <double*>numpy.PyArray_DATA(ob)
    cdef Py_ssize_t i

    for i in range(size):
        data[i] = val

    return ob

cdef object empty_array2(Py_ssize_t size1, Py_ssize_t size2):
    cdef numpy.npy_intp[2] n

    n[0] = size1
    n[1] = size2
    return numpy.PyArray_EMPTY(2, &n[0], numpy.NPY_DOUBLE, 0)

cdef object zeros_array2(Py_ssize_t size1, Py_ssize_t size2):
    cdef numpy.npy_intp[2] n

    n[0] = size1
    n[1] = size2
    return numpy.PyArray_ZEROS(2, &n[0], numpy.NPY_DOUBLE, 0)

cdef object diag_matrix(double[::1] V):
    cdef numpy.npy_intp[2] nn
    cdef double[:,::1] A
    cdef Py_ssize_t i, n

    n = nn[0] = nn[1] = V.shape[0]
    A = o = numpy.PyArray_ZEROS(2, &nn[0], numpy.NPY_DOUBLE, 0)
    for i in range(n):
        A[i,i] = V[i]

    return o

cdef object zeros_array(Py_ssize_t size):
    cdef numpy.npy_intp n = size
    return numpy.PyArray_ZEROS(1, &n, numpy.NPY_DOUBLE, 0)

# cdef inline void swap(double *a, double *b) noexcept nogil:
#     cdef double t=a[0]
#     a[0]=b[0]
#     b[0]=t

cdef double _abs_min(double *a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, v_min = 0

    v_min = fabs(a[0])
    i = 1
    while i < n:
        v = fabs(a[i])
        if v < v_min:
            v_min = v
        i += 1
    return v_min

cdef double _abs_diff_max(double *a, double *b, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, v_max = 0

    v_max = fabs(a[0] - b[0])
    i = 1
    while i < n:
        v = fabs(a[i] - b[i])
        if v > v_max:
            v_max = v
        i += 1
    return v_max

cdef double _mean(double *a, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s/n

cdef double _std(double *a, double mu, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double v, s = 0

    for i in range(n):
        v = a[i] - mu
        s += v*v
    return sqrt(s/n)

cdef double _mad(double *a, double mu, Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += fabs(a[i] - mu)
    return s/n

cdef double quick_select(double *a, Py_ssize_t n): # noexcept nogil:
    cdef Py_ssize_t low, high
    cdef Py_ssize_t median
    cdef Py_ssize_t middle, ll, hh
    cdef double t
    cdef bint is2 = n % 2

    low = 0
    high = n-1
    median = (low + high) // 2
    while 1:
        if high <= low: # One element only
            return a[median]

        if high == low + 1:  # Two elements only
            if a[low] > a[high]:
                t = a[low]; a[low] = a[high]; a[high] = t
            return a[median]

        # Find median of low, middle and high items; swap into position low
        middle = (low + high) // 2
        if a[middle] > a[high]:
            t = a[middle]; a[middle] = a[high]; a[high] = t
        if a[low] > a[high]:
            t = a[low]; a[low] = a[high]; a[high] = t
        if a[middle] > a[low]:
            t = a[middle]; a[middle] = a[low]; a[low] = t

        # Swap low item (now in position middle) into position (low+1)
        # swap(&a[middle], &a[low+1])
        t = a[middle]; a[middle] = a[low+1]; a[low+1] = t

        # Nibble from each end towards middle, swapping items when stuck
        ll = low + 1;
        hh = high;
        while 1:
            ll += 1
            while a[low] > a[ll]:
                ll += 1

            hh -= 1
            while a[hh]  > a[low]:
                hh -= 1

            if hh < ll:
                break

            # swap(&a[ll], &a[hh])
            t = a[ll]; a[ll] = a[hh]; a[hh] = t

        
        # Swap middle item (in position low) back into correct position
        t = a[low]; a[low] = a[hh]; a[hh] = t
        # swap(&a[low], &a[hh])
        
        # Re-set active partition
        if hh <= median:
            low = ll
        if hh >= median:
            high = hh - 1

cdef double _kth_smallest(double *a, Py_ssize_t n, Py_ssize_t k) noexcept nogil:
    cdef Py_ssize_t i, j, l, m
    cdef double x, temp
    cdef double *ai
    cdef double *aj

    l = 0
    m = n-1
    while l < m:
        x = a[k]
        i = l
        j = m
        while 1:
            while a[i] < x:
                i += 1
            while x < a[j]:
                j -= 1
            if i <= j:
                ai = &a[i]
                aj = &a[j]
                temp = ai[0];
                ai[0] = aj[0]
                aj[0] = temp
                # swap(&a[i],&a[j])
                i += 1
                j -= 1
            if i > j:
                break
    
        if j < k:
            l = i
        if k < i: 
            m = j
    
    return a[k]

cdef double _median_1d(double[::1] x): # noexcept nogil:
    cdef Py_ssize_t n2, n = x.shape[0]
    cdef double mv1, mv2
    
    if n % 2:
        n2 = (n-1) // 2
        mv1 = _kth_smallest(&x[0], n, n2)
        return mv1
    else:
        n2 = n // 2
        mv1 = _kth_smallest(&x[0], n, n2)
        mv2 = _kth_smallest(&x[0], n2+1, n2-1)
        return (mv1 + mv2) / 2

cdef double _median_absdev_1d(double[::1] x, double mu):
    cdef Py_ssize_t i, n = x.shape[0]
    cdef double[::1] temp = empty_array(n)

    for i in range(n):
        temp[i] = fabs(x[i] - mu)
    return _median_1d(temp)

cdef void _median_2d(double[:,::1] x, double[::1] y): # noexcept nogil:
    cdef Py_ssize_t i, N = x.shape[0], n = x.shape[1]
    cdef double[::1] temp = empty_array(n)
    
    for i in range(N):
        _move(&temp[0], &x[i,0], n)
        # temp = x[i].copy()
        y[i] = _median_1d(temp)    

cdef void _median_2d_t(double[:,::1] x, double[::1] y): # noexcept nogil:
    cdef Py_ssize_t i, N = x.shape[0], n = x.shape[1]
    cdef double[::1] temp = empty_array(N)
    
    for i in range(n):
        _move_t(&temp[0], &x[0,i], N, n)
        y[i] = _median_1d(temp)

cdef void _median_absdev_2d(double[:,::1] x, double[::1] mu, double[::1] y):
    cdef Py_ssize_t i, j, n = x.shape[0], m = x.shape[1]
    cdef double[::1] temp = empty_array(m)
    cdef double mu_i

    for i in range(n):
        mu_i = mu[i]
        for j in range(m):
            temp[j] = fabs(x[i,j] - mu_i)
        y[i] = _median_1d(temp)

cdef void _median_absdev_2d_t(double[:,::1] x, double[::1] mu, double[::1] y):
    cdef Py_ssize_t i, j, n = x.shape[0], m = x.shape[1]
    cdef double[::1] temp = empty_array(n)
    cdef double mu_j

    for j in range(m):
        mu_j = mu[j]
        for i in range(n):
            temp[i] = fabs(x[i,j] - mu_j)
        y[j] = _median_1d(temp)

cdef double _robust_mean_1d(double[::1] x, double tau): #noexcept nogil:
    cdef Py_ssize_t i, j, q, n = x.shape[0]
    cdef double s, v, mu, std
    cdef double[::1] temp = empty_array(n)

    _move(&temp[0], &x[0], n)
    mu = _median_1d(temp)
    std = _median_absdev_1d(x, mu)

    s = 0
    q = 0
    tau /= 0.6745
    for i in range(n):
        v = x[i]
        if (v <= mu + tau*std) and (v >= mu - tau*std):
            s += v
            q += 1
    return s / q

cdef void _robust_mean_2d_t(double[:,::1] x, double tau, double[::1] y):
    cdef Py_ssize_t i, j, q, n = x.shape[0], m = x.shape[1]
    cdef double[::1] mu = empty_array(m)
    cdef double[::1] std = empty_array(m)
    cdef double s, v, mu_j, std_j

    _median_2d_t(x, mu)
    _median_absdev_2d_t(x, mu, std)

    tau /= 0.6745
    for j in range(m):
        mu_j = mu[j]
        std_j = std[j]
        s = 0
        q = 0
        for i in range(n):
            v = x[i,j]
            if (v <= mu_j + tau*std_j) and (v >= mu_j - tau*std_j):
                s += v
                q += 1
        y[j] = s / q

cdef void _robust_mean_2d(double[:,::1] x, double tau, double[::1] y):
    cdef Py_ssize_t i, j, q, n = x.shape[0], m = x.shape[1]
    cdef double[::1] mu = empty_array(n)
    cdef double[::1] std = empty_array(n)
    cdef double s, v, mu_i, std_i

    _median_2d(x, mu)
    _median_absdev_2d(x, mu, std)

    tau /= 0.6745
    for i in range(n):
        mu_i = mu[i]
        std_i = std[i]
        s = 0
        q = 0
        for j in range(m):
            v = x[i,j]
            if (v <= mu_i + tau*std_i) and (v >= mu_i - tau*std_i):
                s += v
                q += 1
        y[i] = s / q

cdef void _zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu = _mean(a, n)
    cdef double sigma = _std(a, mu, n)

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef void _modified_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu, sigma
    cdef double[::1] aa = empty_array(n)
    # cdef double[::1] ss = empty_array(n)

    _move(&aa[0], &a[0], n)
    mu = _median_1d(aa)
    for i in range(n):
        aa[i] = fabs(aa[i] - mu)
    sigma = _median_1d(aa)
    sigma /= 0.6748
    
    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef void _modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu):
    cdef Py_ssize_t i
    cdef double sigma
    cdef double[::1] aa = empty_array(n)

    # inventory._move(&aa[0], &a[0], n)
    for i in range(n):
        aa[i] = fabs(a[i] - mu)
    sigma = _median_1d(aa)
    sigma /= 0.6748
    
    for i in range(n):
        b[i] = (a[i] - mu) / sigma

cdef void _diff4(double *x, double *y, const Py_ssize_t n4) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n4):
        y[i] = x[i] - 4*x[i+1] + 6*x[i+2] - 4*x[i+3] + x[i+4]

cdef void _diff3(double *x, double *y, const Py_ssize_t n3) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n3):
        y[i] = x[i] - 3*x[i+1] + 3*x[i+2] - x[i+3]

cdef void _diff2(double *x, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    y[0] = 0
    y[n-1] = 0
    for i in range(n-2):
        y[i+1] = x[i] - 2*x[i+1] + x[i+2]

cdef void _diff2w2(double *x, double *w, double *y, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    y[0] = w[0] * x[0] - 2*w[0] * x[1] + w[0] * x[2]
    y[1] = -2*w[0] * x[0] + (4*w[0]+w[1]) * x[1] + \
           (-2*w[0]-2*w[1]) * x[2] + w[1] * x[3]

    for i in range(2,n-2):
        y[i] = w[i-2] * x[i-2] + \
               (-2*w[i-2] - 2*w[i-1]) * x[i-1] + \
               (w[i-2] + 4*w[i-1] + w[i]) * x[i] + \
               (-2*w[i-1]-2*w[i])*x[i+1] + \
               w[i] * x[i+2]

    y[n-2] = w[n-4] * x[n-4] + (-2*w[n-4] -2*w[n-3]) * x[n-3] + \
             (w[n-4] + 4*w[n-3]) * x[n-2] - 2*w[n-4] * x[n-1] 
    y[n-1] = w[n-3]*x[n-3] - 2*w[n-3]*x[n-2] + w[n-3]*x[n-1]

cdef void _diff1(double *x, double *y, const Py_ssize_t n1) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n1):
        y[i] = x[i+1] - x[i]

cdef void _relative_max(double *x, double *y, const Py_ssize_t n) noexcept nogil: 
    cdef Py_ssize_t i
    cdef double v, max_val = 0

    for i in range(n):
        y[i] = x[i]

    for i in range(n):
        v = fabs(y[i])
        if v > max_val:
            max_val = v

    for i in range(n):
        y[i] /= max_val

cdef void _relative_abs_max(double *x, double *y, const Py_ssize_t n) noexcept nogil: 
    cdef Py_ssize_t i
    cdef double v, max_val = 0

    for i in range(n):
        y[i] = fabs(x[i])

    for i in range(n):
        v = y[i]
        if v > max_val:
            max_val = v

    for i in range(n):
        y[i] /= max_val

cdef void _add_to_zeroes(double *a, const Py_ssize_t n, double eps):
    cdef double v 

    for i in range(n):
        v = a[i]
        if abs(v) < eps:
            if v >= 0:
                a[i] = eps
            else:
                a[i] = -eps

def add_to_zeroes(a, eps=1.0e-9):
    cdef double[::1] aa = _asarray(a)
    _add_to_zeroes(&aa[0], aa.shape[0], eps)

def zscore(a, b=None):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0] 
    cdef bint flag = 0

    if b is None:
        bb = b = empty_array(n)
        flag = 1
    else:
        bb = b
    _zscore(&aa[0], &bb[0], aa.shape[0])
    return b

def modified_zscore2(a, b=None):
    cdef double[:,::1] aa = _asarray(a)
    cdef double[:,::1] bb
    cdef Py_ssize_t i, n = aa.shape[0], m = aa.shape[1]
    cdef bint flag = 0
    if b is None:
        bb = b = empty_array2(n,m)
        flag = 1
    else:
        bb = b
    for i in range(n):
        _modified_zscore(&aa[i,0], &bb[i,0], m)
    if flag:
        return b
    else:
        return _asarray(bb)

def modified_zscore(a, b=None, mu=None):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0]
    cdef double d_mu
    cdef bint flag = 0
    if b is None:
        bb = b = empty_array(n)
        flag = 1
    if mu is None:
        _modified_zscore(&aa[0], &bb[0], n)
    else:
        d_mu = mu
        _modified_zscore_mu(&aa[0], &bb[0], n, d_mu)
    if flag:
        return b
    else:
        return _asarray(bb)

def diff4(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-4)
        flag = 1
    _diff4(&a[0], &b[0], n-4)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff3(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-3)
        flag = 1
    _diff3(&a[0], &b[0], n-3)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff2(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n)
        flag = 1
    _diff2(&a[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff2w2(double[::1] a, double[::1] w, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        b = empty_array(n)
        flag = 1
    _diff2w2(&a[0], &w[0], &b[0], n)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def diff1(double[::1] a, double[::1] b=None):
    cdef Py_ssize_t n = a.shape[0] 
    cdef bint flag = 0
    if b is None:
        b = empty_array(n-1)
        flag = 1
    _diff1(&a[0], &b[0], n-1)
    if flag:
        return b.base
    else:
        return np.asarray(b)

def relative_max(a, b=None):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0]
    # cdef bint flag = 0
    if b is None:
        bb = b = empty_array(n)
        flag = 1
    else:
        bb = b
    _relative_max(&aa[0], &bb[0], n)
    return b

def relative_abs_max(a, b=None, fix_ends=True):
    cdef double[::1] aa = _asarray(a)
    cdef double[::1] bb
    cdef Py_ssize_t n = a.shape[0]
    cdef bint flag = 0
    if b is None:
        bb = b = empty_array(n)
        flag = 1
    else:
        bb = b
    _relative_abs_max(&aa[0], &bb[0], n)
    if fix_ends:
        min_b = np.min(b)
        b[0] = b[-1] = min_b
    return b

def median(a, axis=None, out=None, overwrite_input=False):
    # can't be reasonably be implemented in terms of percentile as we have to
    # call mean to not break astropy
    # a = np.asanyarray(a)
    a = _asarray(a)

    # Set the partition indexes
    if axis is None:
        sz = a.size
    else:
        sz = a.shape[axis]
    if sz % 2 == 0:
        szh = sz // 2
        kth = [szh - 1, szh]
    else:
        kth = [(sz - 1) // 2]

    # We have to check for NaNs (as of writing 'M' doesn't actually work).
    # supports_nans = np.issubdtype(a.dtype, np.inexact) or a.dtype.kind in 'Mm'
    # if supports_nans:
    #     kth.append(-1)

    if not overwrite_input:
        a = a.copy()
    if axis is None:
        part = a.ravel()
        part.partition(kth)
    else:
        a.partition(kth, axis=axis)
        part = a

    if part.shape == ():
        # make 0-D arrays work
        return part.item()
    if axis is None:
        axis = 0

    indexer = [slice(None)] * part.ndim
    index = part.shape[axis] // 2
    if part.shape[axis] % 2 == 1:
        # index with slice to allow mean (below) to work
        indexer[axis] = slice(index, index + 1)
    else:
        indexer[axis] = slice(index - 1, index + 1)
    indexer = tuple(indexer)

    # Use mean in both odd and even case to coerce data type,
    # using out array if needed.
    rout = part[indexer].mean(axis=axis, out=out)
    # if supports_nans and sz > 0:
    #     # If nans are possible, warn and replace by nans like mean would.
    #     rout = np.lib._utils_impl._median_nancheck(part, rout, axis)

    return rout

def median_1d(x, copy=True):
    if copy:
        xx = x.copy()
    else:
        xx = x
    return _median_1d(_asarray(xx))

def median_absdev_1d(x, mu):
    return _median_absdev_1d(_asarray(x), mu)

def median_2d_t(x):
    y = empty_array(x.shape[1])
    _median_2d_t(_asarray(x), y)
    return y

def median_2d(x):
    y = empty_array(x.shape[0])
    _median_2d(_asarray(x), y)
    return y

def median_absdev_2d(x, mu):
    y = empty_array(x.shape[0])
    _median_absdev_2d(_asarray(x), mu, y)
    return y

def median_absdev_2d_t(x, mu):
    y = empty_array(x.shape[1])
    _median_absdev_2d_t(_asarray(x), mu, y)
    return y

def robust_mean_1d(x, tau):
    return _robust_mean_1d(_asarray(x), tau)

# def robust_mean_1d(x, tau):
#     x = _asarray(x)

#     mu = median(x)
#     std = median(abs(x - mu))

#     tau /= 0.6745
#     tau_std = tau * std
#     return  x[(v <= mu + tau_std) & (v >= mu - tau_std)].mean()

def robust_mean_2d(x, tau):
    y = empty_array(x.shape[0])
    _robust_mean_2d(_asarray(x), tau, y)
    return y

def robust_mean_2d_t(x, tau):
    y = empty_array(x.shape[1])
    _robust_mean_2d_t(_asarray(x), tau, y)
    return y

# cdef void _covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S) noexcept nogil:
#     cdef Py_ssize_t i, j
#     cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
#     cdef double s, loc_i, loc_j
#     #
#     for i in range(n):
#         loc_i = loc[i]
#         for j in range(n):
#             loc_j = loc[j]
#             s = 0
#             for k in range(N):
#                 s += (X[k,i] - loc_i) * (X[k,j] - loc_j)
#             S[i,j] = s / N

# def covariance_matrix(X, loc=None, S=None):
#     X = _asarray(X)
#     n = X.shape[1]
#     if S is None:
#         S = empty_array2(n, n)
#     else:
#         S = _asarray(S)
#         if S.shape[0] != n and S.shape[1] != n:
#             raise TypeError(f"ivalid shape of S: {S.shape}")
#     if loc is None:
#         loc = zeros_array(n)
#     else:
#         loc = _asarray(loc)
#     _covariance_matrix(X, loc, S)
#     return S

# cdef void _covariance_matrix_weighted(double[:, ::1] X, double[::1] W, 
#                                       double[::1] loc, double[:,::1] S) noexcept nogil:
#     cdef Py_ssize_t i, j
#     cdef Py_ssize_t n = X.shape[1], N = X.shape[0]
#     cdef double s, loc_i, loc_j
#     #
#     for i in range(n):
#         loc_i = loc[i]
#         for j in range(n):
#             loc_j = loc[j]
#             s = 0
#             for k in range(N):
#                 s += W[k] * (X[k,i] - loc_i) * (X[k,j] - loc_j)
#             S[i,j] = s

# cdef void _covariance_matrix_weighted(
#             double *X, const double *W, const double *loc, double *S, 
#             const Py_ssize_t n, const Py_ssize_t N) noexcept nogil:

#     cdef Py_ssize_t i, j, k
#     cdef double s, loc_i, loc_j
#     cdef double *X_ki
#     cdef double *X_kj
#     cdef double *S_i
#     cdef double *S_j

#     S_i = S_j = S
#     for i in range(n):
#         loc_i = loc[i]
#         for j in range(i, n):
#             loc_j = loc[j]
#             X_kj = X + j
#             X_ki = X + i

#             s = 0
#             for k in range(N):
#                 s += W[k] * (X_ki[0] - loc_i) * (X_kj[0] - loc_j)
#                 X_ki += n
#                 X_kj += n

#             S_i[j] = S_j[i] = s
#             S_j += n
#         S_i += n
            
# def covariance_matrix_weighted(X, W, loc=None, S=None):
#     X = _asarray(X)
#     n = X.shape[1]
#     W = _asarray(W)
#     if S is None:
#         S = empty_array2(n, n)
#     else:
#         S = _asarray(S)
#         if S.shape[0] != n and S.shape[1] != n:
#             raise TypeError(f"ivalid shape of S: {S.shape}")
#     if loc is None:
#         loc = zeros_array(n)
#     else:
#         loc = _asarray(loc)
#     _covariance_matrix_weighted(X, W, loc, S)
#     return S

cdef class RingArray:
    #
    def __init__(self, Py_ssize_t n):
        self.data = zeros_array(n)
        self.size = n
        self.index = 0
    #
    cpdef add(self, double val):
        self.data[self.index] = val
        self.index += 1
        if self.index == self.size:
            self.index = 0
    #
    cpdef mad(self):
        cdef double mu_val, mad_val

        mu_val = _mean(&self.data[0], self.size)
        mad_val = _mad(&self.data[0], mu_val, self.size)
        return mad_val
    #
    # cpdef median(self):
    #     return median_1d(self.data, True)
            

cdef _inverse_matrix(double[:,::1] AM, double[:,::1] IM):
    """
    Returns the inverse of the passed in matrix.
        :param A: The matrix to be inversed

        :return: The inverse of the matrix A
    """
    # Section 1: Make sure A can be inverted.
    # check_squareness(A)
    # check_non_singular(A)

    # Section 2: Make copies of A & I, AM & IM, to use for row operations
    cdef Py_ssize_t fd, i, j, n = len(AM)
    cdef double fdScaler, crScaler
    cdef double *AM_fd
    cdef double *AM_i
    cdef double *IM_fd
    cdef double *IM_i

    _clear(&IM[0,0], n*n)
    for i in range(n):
        IM[i,i] = 1

    # Section 3: Perform row operations
    for fd in range(n): # fd stands for focus diagonal
        # FIRST: scale fd row with fd inverse.
        AM_fd = &AM[fd,0]
        IM_fd = &IM[fd,0]
        fdScaler = 1.0 / AM[fd,fd]
        for j in range(n): # Use j to indicate column looping.
            AM_fd[j] *= fdScaler
            IM_fd[j] *= fdScaler
        # SECOND: operate on all rows except fd row as follows:
        for i in range(n): # *** skip row with fd in it.
            if i == fd:
                continue
            AM_i = &AM[i,0]
            IM_i = &IM[i,0]
            crScaler = AM[i,fd] # cr stands for "current row".
            for j in range(n): # cr - crScaler * fdRow, but one element at a time.
                AM_i[j] -= crScaler * AM_fd[j]
                IM_i[j] -= crScaler * IM_fd[j]

    # Section 4: Make sure that IM is an inverse of A within the specified tolerance
    # if check_matrix_equality(I,matrix_multiply(A,IM),tol):
    #     return IM
    # else:
    #     raise ArithmeticError("Matrix inverse out of tolerance.")

def inverse_matrix(A, copy=1):
    n, m = A.shape
    if copy:
        AM = A.copy()
    else:
        AM = A
    IM = empty_array2(n, n)
    _inverse_matrix(AM, IM)
    return IM

cdef double _norm2(double[::1] a):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s, v

    s = 0
    for i in range(n):
        v = a[i]
        s += v*v
    return sqrt(s)

def norm2(a):
    return _norm2(a)

def scale_min(x, alpha=0.01):
    if x < 0:
        return (1+alpha)*x
    else:
        return (1-alpha)*x

def scale_max(x, alpha=0.01):
    if x < 0:
        return (1.0-alpha)*x
    else:
        return (1.0+alpha)*x

def array_bounds(a, pad=0.1):
    """Нахождение нижней и верхней границы, между которыми заключены значения массива.
    Параметры:
        a: 1-мерный массив
        pad: доля от минимиального и максимального значения, которые будут добавляться слева к `min(a)` и справа к `max(a)`, соответственно.
    """
    return scale_min(a.min(), pad), scale_max(a.max(), pad)
