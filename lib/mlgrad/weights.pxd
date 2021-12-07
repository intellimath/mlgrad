# cython: language_level=3

cimport cython

from mlgrad.func cimport Func
from mlgrad.risk cimport ERisk, Risk, Functional
from mlgrad.avragg cimport Average
from mlgrad.model cimport Model

cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

cdef inline float array_min(float[::1] arr):
    cdef Py_ssize_t i, N = arr.shape[0]
    cdef float v, min_val = arr[0]

    for i in range(N):
        v = arr[i]
        if v < min_val:
            min_val = v

    return min_val

cdef inline float sum_memoryview(float[::1] X):
    cdef float S
    cdef Py_ssize_t i, m = X.shape[0]
    
    S = 0
    for i in range(m):
        S += X[i]
    return S

cdef inline void mult_scalar_memoryview(float[::1] X, float c):
    cdef Py_ssize_t i, m = X.shape[0]

    for i in range(m):
        X[i] *= c
        
cdef inline void normalize_memoryview(float[::1] X):
    cdef float S, c
    cdef Py_ssize_t i, m = X.shape[0]

    S = 0
    for i in range(m):
        S += X[i]
    c = 1/S
    
    for i in range(m):
        X[i] *= c        

cdef class Weights(object):
    #
    cdef public float[::1] weights
    #
    cdef init(self)
    cdef eval_weights(self)
    cdef float[::1] get_weights(self)
    cdef float get_qvalue(self)
    cdef set_param(self, name, val)

@cython.final   
cdef class ArrayWeights(Weights):
    pass

@cython.final
cdef class ConstantWeights(Weights):
    pass

@cython.final
cdef class RWeights(Weights):
    cdef readonly float[::1] lval_all
    cdef public Func func
    cdef public Risk risk
    cdef bint normalize

@cython.final
cdef class MWeights(Weights):
    cdef readonly float[::1] lval_all
    cdef float best_u
    cdef public Average average
    cdef public Risk risk
    cdef bint first_time, normalize, u_only, use_best_u

# @cython.final
# cdef class SWMWeights(Weights):
#     cdef float[::1] lval_all
#     cdef public Average average
#     cdef public Risk risk
#     cdef bint first_time, normalize, u_only

# @cython.final
# cdef class WMWeights(Weights):
#     cdef float[::1] lval_all
#     cdef public Average average
#     cdef public Risk risk
#     cdef bint first_time, normalize, u_only
