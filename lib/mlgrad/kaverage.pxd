# coding: utf-8

# cython: language_level=3

cimport cython

from mlgrad.func cimport Func, Square
from libc.string cimport memcpy, memset

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef float max_float, min_float
cdef Func square_func

cdef class KAverage:
    cdef public Func func
    cdef public float[::1] u, u_prev, u_min
    cdef public Py_ssize_t q
    cdef public float tol
    cdef public int K, n_iter
    cdef float qval, qval_min
    cdef public list qvals
#     cdef Py_ssize_t[::1] j_min
#     cdef float[::1] d_min
    
    cdef _init_u(self, float[::1] Y)
    cdef _evaluate_classes(self, float[::1] Y, Py_ssize_t[::1] J)
    cdef _evaluate_distances(self, float[::1] Y, float[::1] D)
    cdef float _evaluate_qval(self, float[::1] Y)
    cdef _fit(self, float[::1] Y)

# cdef class ScalarKMeans(KAverage):
    
#     cdef public float[::1] u
#     cdef public Py_ssize_t q
#     cdef public float tol
#     cdef public int n_iter
    
#     cdef _evaluate_classes(self, float[::1] Y, int[::1] J)
#     cdef _evaluate_distances(self, float[::1] Y, float[::1] D)
#     cdef _fit(self, float[::1] Y)
