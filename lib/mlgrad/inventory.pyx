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
from openmp cimport omp_get_num_procs
from cpython.object cimport PyObject
 
cdef int num_threads = 2 #omp_get_num_procs()
# if num_procs >= 4:
#     num_procs /= 2
# else:
#     num_procs = 2

cdef int get_num_threads() nogil:
    return num_threads

cdef void set_num_threads(int num) nogil:
    num_threads = num
    
cdef void _clear(double *to, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = 0

cdef void clear(double[::1] to) nogil:
    _clear(&to[0], <const size_t>to.shape[0])

cdef void _clear2(double *to, const size_t n, const size_t m) nogil:
    cdef size_t i
    for i in range(n*m):
        to[i] = 0
        
cdef void clear2(double[:,::1] to) nogil:
    _clear2(&to[0,0], <const size_t>to.shape[0], <const size_t>to.shape[1])
    
cdef void _fill(double *to, const double c, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = c

cdef void fill(double[::1] to, const double c) nogil:
    _fill(&to[0], c, <const size_t>to.shape[0])
        
cdef void _move(double *to, const double *src, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = src[i]

cdef void move(double[::1] to, double[::1] src) nogil:
    _move(&to[0], &src[0], to.shape[0])

cdef void move2(double[:, ::1] to, double[:,::1] src) nogil:
    _move(&to[0,0], &src[0,0], to.shape[0] * to.shape[1])

cdef void move3(double[:,:,::1] to, double[:,:,::1] src) nogil:
    _move(&to[0,0,0], &src[0,0,0], to.shape[0] * to.shape[1] * to.shape[2])
    
cdef double _conv(const double *a, const double *b, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[i]
    return s

cdef double conv(double[::1] a, double[::1] b) nogil:
    return _conv(&a[0], &b[0], a.shape[0])

cdef void _add(double *a, const double *b, const size_t n) nogil:
    cdef size_t i

    for i in range(n):
        a[i] += b[i]

cdef void add(double[::1] a, double[::1] b) nogil:
    _add(&a[0], &b[0], a.shape[0])

cdef void add2(double[:,::1] a, double[:,::1] b) nogil:
    _add(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _sub(double *a, const double *b, const size_t n) nogil:
    cdef size_t i

    for i in range(n):
        a[i] -= b[i]

cdef void sub(double[::1] a, double[::1] b) nogil:
    _sub(&a[0], &b[0], a.shape[0])
    
cdef double _sum(const double *a, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s

cdef double sum(double[::1] a) nogil:
    return _sum(&a[0], a.shape[0])

cdef void _mul_const(double *a, const double c, const size_t n) nogil:
    cdef size_t i

    for i in range(n):
        a[i] *= c

cdef void mul_const(double[::1] a, const double c) nogil:
    _mul_const(&a[0], c, a.shape[0])

cdef void mul_const2(double[:,::1] a, const double c) nogil:
    _mul_const(&a[0,0], c, a.shape[0] * a.shape[1])

cdef void mul_const3(double[:,:,::1] a, const double c) nogil:
    _mul_const(&a[0,0,0], c, a.shape[0] * a.shape[1] * a.shape[2])
    
cdef void _mul_add(double *a, const double *b, double c, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] += c * b[i]

cdef void mul_add(double[::1] a, double[::1] b, double c) nogil:
    _mul_add(&a[0], &b[0], c, a.shape[0])

cdef void mul_add2(double[:,::1] a, double[:,::1] b, double c) nogil:
    _mul_add(&a[0,0], &b[0,0], c, a.shape[0] * a.shape[1])
    
cdef void _mul_set(double *a, const double *b, double c, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] = c * b[i]

cdef void mul_set(double[::1] a, double[::1] b, double c) nogil:
    _mul_set(&a[0], &b[0], c, a.shape[0])
        
cdef void _mul(double *a, const double *b, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] *= b[i]

cdef void mul(double[::1] a, double[::1] b) nogil:
    _mul(&a[0], &b[0], a.shape[0])

cdef void mul2(double[:,::1] a, double[:,::1] b) nogil:
    _mul(&a[0,0], &b[0,0], a.shape[0] * a.shape[1])
    
cdef void _multiply(double *a, const double *b, const double *c, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] = b[i] * c[i]

cdef void multiply(double[::1] a, double[::1] b, double[::1] c) nogil:
    _multiply(&a[0], &b[0], &c[0], a.shape[0])
    
cdef void _matdot(double *output, double *M, const double *X, 
                    const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef double s
    cdef double *Mj = M

    for j in range(n_output):
        s = 0
        for i in range(n_input):
            s += Mj[i] * X[i];
        output[j] = s
        Mj += n_input

cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) nogil:
    _matdot(&output[0], &M[0,0], &X[0], X.shape[0], output.shape[0])
        
cdef void _matdot2(double *output, double *M, const double *X, 
                   const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef double s
    cdef double *Mj = M;

    for j in range(n_output):
        s = Mj[0]
        Mj += 1
        for i in range(n_input):
            s += Mj[i] * X[i]
        output[j] = s
        Mj += n_input

cdef void matdot2(double[::1] output, double[:,::1] M, double[::1] X) nogil:
    _matdot2(&output[0], &M[0,0], &X[0], <const size_t>X.shape[0], <const size_t>M.shape[0])
        
cdef void _mul_add_arrays(double *a, double *M, const double *ss, 
                          const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef double *Mj = M;
    cdef double sx

    for j in range(n_output):
        Mj += 1
        sx = ss[j]
        for i in range(n_input):
            a[i] += sx * Mj[i]
        Mj += n_input

cdef void mul_add_arrays(double[::1] a, double[:,::1] M, double[::1] ss) nogil:
    _mul_add_arrays(&a[0], &M[0,0], &ss[0], <const size_t>(a.shape[0]), <const size_t>(M.shape[0]))
        
cdef void _mul_grad(double *grad, const double *X, const double *ss, 
                    const size_t n_input, const size_t n_output) nogil:
    cdef size_t i, j
    cdef double *G = grad
    cdef double sx
    
    for j in range(n_output):
        sx = ss[j]
        G[0] = sx
        G += 1
        for i in range(n_input):
            G[i] = sx * X[i]
        G += n_input

cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) nogil:
    _mul_grad(&grad[0,0], &X[0], &ss[0], <const size_t>X.shape[0], <const size_t>grad.shape[0])

cdef void _normalize(double *a, const size_t n) nogil:
    cdef size_t i
    cdef double S

    S = 0
    for i in range(n):
        S += a[i]

    for i in range(n):
        a[i] /= S

cdef void normalize(double[::1] a) nogil:
    _normalize(&a[0], a.shape[0])

cdef void scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) nogil:
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
        size_t N = X.shape[0]
        size_t n = X.shape[1]
        size_t i, j, k
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

cdef void scatter_matrix(double[:,::1] X, double[:,::1] S) nogil:
    """
    Вычисление ковариационной матрицы
    Вход:
       X: матрица (N,n)
    Результат:
       S: матрица (n,n):
          S = (1/N) X.T @ X
    """
    cdef:
        size_t N = X.shape[0]
        size_t n = X.shape[1]
        size_t i, j, k
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

cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) nogil:
    """
    Взвешенная сумма строк матрицы:
    Вход:
       X: матрица (N,n)
       W: массив весов (N)
       Y: массив (N) - результат:
          Y[i] = W[0] * X[0,:] + ... + W[N-1] * X[N-1,:]
    
    """
    cdef:
        size_t N = X.shape[0]
        size_t n = X.shape[1]
        size_t i, k
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
