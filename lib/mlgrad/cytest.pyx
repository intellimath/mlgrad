# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

from cython.parallel cimport parallel, prange

import numpy as np

cdef void fill(float *X, float v, int n):
    cdef int i
    for i in range(n):
        X[i] = v

cdef double conv_parallel(double[::1] A, double[::1] B) nogil:
    cdef Py_ssize_t i, n = A.shape[0]
    cdef double c

    c = 0
    for i in prange(n, nogil=True, schedule='static'):
        c += A[i] * B[i]

    return c

cdef double conv(double[::1] A, double[::1] B) nogil:
    cdef int i, n = A.shape[0]
    cdef double c

    c = 0
    for i in range(n):
        c += A[i] * B[i]

    return c        

def dot(double[:,::1] A, double[::1] B, double[::1] C):
    cdef int i, j, n = A.shape[0], m = B.shape[0]
    cdef double c

    if A.shape[1] != m:
        raise RuntimeError("A.shape[1] != B.shape[0]")

#     fill(C, 0)
    for i in range(n):
        c = 0
        for j in range(m):
            c += A[i,j] * B[j]
        C[i] = c

    return C.base

def dot_parallel(double[:,::1] A, double[::1] B, double[::1] C):
    cdef int i, j, n = A.shape[0], m = B.shape[0]
    cdef double c
    
    if A.shape[1] != m:
        raise RuntimeError("A.shape[1] != B.shape[0]")

#     fill(C, 0)
    with nogil:
        for i in prange(n, schedule='static'):
            C[i] = conv(A[i], B)
#             for j in range(m):
#                 C[i] += A[i,j] * B[j]
