# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void matrix_dot(double[:,::1] A, double[::1] x, double[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef double v
    
    for j in range(n):
        v = 0
        for i in range(m):
            v += A[j,i] * x[i]
        y[j] = v

cdef inline void matrix_dot_t(double[:,::1] A, double[::1] x, double[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef double v
    
    for i in range(m):
        v = 0
        for j in range(n):
            v += A[j,i] * x[j]
        y[i] = v

cdef class FuncMulti:
    #cdef bint all
    cdef double evaluate(self, double[::1] param)
    cdef void gradient(self, double[::1] param, double[::1] grad)

@cython.final
cdef class Power(FuncMulti):
    #
    cdef double p
    #

@cython.final
cdef class Square(FuncMulti):
    pass

@cython.final
cdef class Absolute(FuncMulti):
    pass

@cython.final
cdef class SquareForm(FuncMulti):
    cdef double[:,::1] matrix

@cython.final
cdef class Rosenbrok(FuncMulti):
    pass

@cython.final
cdef class Himmelblau(FuncMulti):
    pass