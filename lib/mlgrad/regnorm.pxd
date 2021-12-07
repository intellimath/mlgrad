# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void matrix_dot(float[:,::1] A, float[::1] x, float[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef float v
    
    for j in range(n):
        v = 0
        for i in range(m):
            v += A[j,i] * x[i]
        y[j] = v

cdef inline void matrix_dot_t(float[:,::1] A, float[::1] x, float[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef float v
    
    for i in range(m):
        v = 0
        for j in range(n):
            v += A[j,i] * x[j]
        y[i] = v

cdef class FuncMulti:
    #cdef bint all
    cdef float evaluate(self, float[::1] param)
    cdef void gradient(self, float[::1] param, float[::1] grad)

@cython.final
cdef class PowerNorm(FuncMulti):
    #
    cdef float p
    #

@cython.final
cdef class SquareNorm(FuncMulti):
    pass

@cython.final
cdef class AbsoluteNorm(FuncMulti):
    pass

@cython.final
cdef class SquareForm(FuncMulti):
    cdef float[:,::1] matrix

@cython.final
cdef class Rosenbrok(FuncMulti):
    pass

@cython.final
cdef class Himmelblau(FuncMulti):
    pass