# coding: utf-8 

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=False
# cython: initializedcheck=False
# cython: unraisable_tracebacks=True  

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


cdef void fa_fill(double *to, const double c, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = c

        
cdef void fa_move(double *to, const double *src, const size_t n) nogil:
    cdef size_t i
    for i in range(n):
        to[i] = src[i]
        
cdef double fa_conv(const double *a, const double *b, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i] * b[i]
    return s

cdef double fa_sum(const double *a, const size_t n) nogil:
    cdef size_t i
    cdef double s = 0

    for i in range(n):
        s += a[i]
    return s

cdef void fa_mul_const(double *a, const double c, const size_t n) nogil:
    cdef size_t i

    for i in range(n):
        a[i] *= c

cdef void fa_mul_add_array(double *a, const double *b, double c, const size_t n) nogil:
    cdef size_t i
    
    for i in range(n):
        a[i] += c * b[i]

# cdef void fa_matdot(double *output, double *M, const double *X, 
#                     const size_t n_input, const size_t n_output) nogil:
#     cdef size_t i, j
#     cdef double s
#     cdef double *Mj = M

#     for j in range(n_output):
#         s = 0
#         for i in range(n_input):
#             s += Mj[i] * X[i];
#         output[j] = s
#         Mj += n_input

cdef void fa_matdot2(double *output, double *M, const double *X, 
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

cdef void fa_mul_add_arrays(double *a, double *M, const double *ss, 
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

cdef void fa_mul_grad(double *grad, const double *X, const double *ss, 
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
