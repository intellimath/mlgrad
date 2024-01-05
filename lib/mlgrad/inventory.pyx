# coding: utf-8 

# The MIT License (MIT)
#
# Copyright (c) <2015-2021> <Shibzukhov Zaur, szport at gmail dot com>
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

from cython.parallel cimport parallel, prange
 
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
    
cdef double _conv(const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[i]
    return s

cdef double conv(double[::1] a, double[::1] b) noexcept nogil:
    return _conv(&a[0], &b[0], a.shape[0])

cdef void _add(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] += b[i]

cdef void add(double[::1] a, double[::1] b) noexcept nogil:
    _add(&a[0], &b[0], a.shape[0])

cdef void add2(double[:,::1] a, double[:,::1] b) noexcept nogil:
    _add(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _sub(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] -= b[i]

cdef void sub(double[::1] a, double[::1] b) noexcept nogil:
    _sub(&a[0], &b[0], a.shape[0])
    
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
    
cdef void _mul(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        a[i] *= b[i]

cdef void mul(double[::1] a, double[::1] b) noexcept nogil:
    _mul(&a[0], &b[0], a.shape[0])

cdef void mul2(double[:,::1] a, double[:,::1] b) noexcept nogil:
    _mul(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _multiply(double *a, const double *b, const double *c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    
    for i in range(n):
        a[i] = b[i] * c[i]

cdef void multiply(double[::1] a, double[::1] b, double[::1] c) noexcept nogil:
    _multiply(&a[0], &b[0], &c[0], a.shape[0])


cdef double dot1(double[::1] a, double[::1] b) noexcept nogil:
    return _dot1(&a[0], &b[0], a.shape[0])

cdef double _dot1(const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = a[0]

    a += 1
    for i in range(n):
        s += a[i] * b[i]
    return s

cdef double dot(double[::1] a, double[::1] b) noexcept nogil:
    return _dot(&a[0], &b[0], a.shape[0])

cdef double _dot(const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[i]
    return s

cdef double _dot_t(const double *a, double *b, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[0]
        b += m
    return s

cdef void _matdot(double *output, double *M, const double *X, 
                    const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t j

    # for j in prange(n_output, schedule='static', nogil=True, num_threads=num_procs):
    for j in range(n_output):
        output[j] = _dot(M + j * n_input, X, n_input)

cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil:
    _matdot(&output[0], &M[0,0], &X[0], X.shape[0], output.shape[0])
        
cdef void _matdot2(double *output, double *M, const double *X, 
                   const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t j
    cdef double *Mj

    # for j in prange(n_output, schedule='static', nogil=True, num_threads=num_procs):
    for j in range(n_output):
        Mj = M + j * (n_input+1)
        output[j] = Mj[0] + _dot(&Mj[1], X, n_input)

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
        G += n_input

cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) noexcept nogil:
    _mul_grad(&grad[0,0], &X[0], &ss[0], <const Py_ssize_t>X.shape[0], <const Py_ssize_t>grad.shape[0])

cdef void _normalize(double *a, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    cdef double S

    S = 0
    for i in range(n):
        S += a[i]

    for i in range(n):
        a[i] /= S

cdef void normalize(double[::1] a) noexcept nogil:
    _normalize(&a[0], a.shape[0])

cdef void scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil:
    """
    Вычисление взвешенной ковариационной матрицы
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
    Результат:
       S: матрица (n,n):
          S = (1/N) (W[0] * outer(X[0,:],X[0,:]) + ... + W[N-1] * outer(X[N-1,:],X[N-1,:]))
    """
    cdef:
        Py_ssize_t N = X.shape[0]
        Py_ssize_t n = X.shape[1]
        Py_ssize_t i, j, k
        double s
        double *Xk
        double *ss

    for i in range(n):
        ss = &S[0,0]
        for j in range(n):
            Xk = &X[0,0]
            s = 0
            for k in range(N):
                s += W[k] * Xk[i] * Xk[j]
                Xk += n
            ss[j] = s
        ss += n
    
    # for k in range(N):
    #     wk = W[k]
    #     Xk = &X[k,0]
    #     ss = &S[0,0]
    #     for i in range(n):
    #         v = wk * Xk[i]
    #         for j in range(n):
    #             ss[j] += v * Xk[j]
    #         ss += n

cdef void scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil:
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

    for i in range(n):
        ss = &S[0,0]
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
        Xk = &X[0,0]
        for k in range(N):
            wk = W[k]
            y += wk * Xk[i]
            Xk += n
        yy[i] = y
