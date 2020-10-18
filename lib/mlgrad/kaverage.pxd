# coding: utf-8

# cython: language_level=3

cimport cython

from mlgrad.func cimport Func, Square
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

cdef double max_double, min_double
cdef Func square_func

cdef class KAverage:
    cdef public Func func
    cdef public double[::1] u, u_prev, u_min
    cdef public Py_ssize_t q
    cdef public double tol
    cdef public int K, n_iter
    cdef double qval, qval_min
    cdef public list qvals
#     cdef Py_ssize_t[::1] j_min
#     cdef double[::1] d_min
    
    cdef _init_u(self, double[::1] Y)
    cdef _evaluate_classes(self, double[::1] Y, Py_ssize_t[::1] J)
    cdef _evaluate_distances(self, double[::1] Y, double[::1] D)
    cdef double _evaluate_qval(self, double[::1] Y)
    cdef _fit(self, double[::1] Y)

# cdef class ScalarKMeans(KAverage):
    
#     cdef public double[::1] u
#     cdef public Py_ssize_t q
#     cdef public double tol
#     cdef public int n_iter
    
#     cdef _evaluate_classes(self, double[::1] Y, int[::1] J)
#     cdef _evaluate_distances(self, double[::1] Y, double[::1] D)
#     cdef _fit(self, double[::1] Y)
