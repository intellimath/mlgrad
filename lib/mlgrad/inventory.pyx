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


cdef void init_rand() noexcept nogil:
    srand(time(NULL))

cdef long rand(long N) noexcept nogil:
    return stdlib_rand() % N

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

cdef void _mul_const(double *a, double *b, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] = b[i] * c

cdef void mul_const(double[::1] a, double[::1] b, const double c) noexcept nogil:
    _mul_const(&a[0], &b[0], c, a.shape[0])

cdef void _imul_const(double *a, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] *= c

cdef void imul_const(double[::1] a, const double c) noexcept nogil:
    _imul_const(&a[0], c, a.shape[0])

cdef void imul_const2(double[:,::1] a, const double c) noexcept nogil:
    _imul_const(&a[0,0], c, a.shape[0] * a.shape[1])

cdef void imul_const3(double[:,:,::1] a, const double c) noexcept nogil:
    _imul_const(&a[0,0,0], c, a.shape[0] * a.shape[1] * a.shape[2])

cdef void _imul_add(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i

    for i in range(n):
        a[i] += c * b[i]

cdef void imul_add(double[::1] a, double[::1] b, const double c) noexcept nogil:
    _imul_add(&a[0], &b[0], c, a.shape[0])

cdef void imul_add2(double[:,::1] a, double[:,::1] b, const double c) noexcept nogil:
    _imul_add(&a[0,0], &b[0,0], c, a.shape[0] * a.shape[1])

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

cdef double linear_func(double[::1] a, double[::1] x) noexcept nogil:
    return _linear_func(&a[0], &x[0], a.shape[0])

cdef double _linear_func(const double *a, const double *x, const Py_ssize_t n) noexcept nogil:
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

cdef void moving_dot(double[::1] b, double[::1] a, double[::1] x) noexcept nogil:
    _moving_dot(&b[0], &a[0], &x[0], a.shape[0], x.shape[0])

cdef void _moving_dot(double *b, const double *a, const double *x, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i,j
    cdef double s

    for j in range(n-m):
        s = 0
        for i in range(m):
            s += a[i] * x[i]
        b[j] = s
        a += 1

cdef double dot_t(double[::1] a, double[:,::1] b) noexcept nogil:
    return _dot_t(&a[0], &b[0,0], a.shape[0], b.shape[0])

cdef double _dot_t(const double *a, double *b, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[0]
        b += m
    return s

cdef void _matdot_sparse(double *output, double *M, const double *X, 
                    const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef double *M_j = M
    cdef double s, v

    # for j in prange(n_output, schedule='static', nogil=True, num_threads=num_procs):
    for j in range(n_output):
        s = 0
        for i in range(n_input):
            v = M_j[i]
            if v == 0:
                continue
            s += v * X[i]
        output[j] = s
        M_j += n_input

cdef void matdot_sparse(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil:
    _matdot_sparse(&output[0], &M[0,0], &X[0], X.shape[0], output.shape[0])

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

cdef _mat_diagonal(double *diag, double* mat, Py_ssize_t d, Py_ssize_t n):
    cdef Py_ssize_t i = 0, j = d
    if d >= 0:
        j = d
    else:
        d = -d
        j = d * n
    for i in range(n - d):
        diag[i] = mat[j]
        j += n+1

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

    if max_val == 0:
        max_val = 1

    for i in range(n):
        y[i] /= max_val

cdef void _add_to_zeros(double *a, const Py_ssize_t n, double eps):
    cdef double v

    for i in range(n):
        v = a[i]
        if abs(v) < eps:
            if v >= 0:
                a[i] = eps
            else:
                a[i] = -eps

def add_to_zeros(a, eps=1.0e-9):
    cdef double[::1] aa = _asarray(a)
    _add_to_zeros(&aa[0], aa.shape[0], eps)

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
        b[0] = b[n-1] = min_b
    return b


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

include "inventory_array.pyx"
include "inventory_matrix.pyx"
include "inventory_statistics.pyx"
include "inventory_diff.pyx"