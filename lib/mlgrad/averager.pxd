# cython: language_level=3

cimport cython
from libc.string cimport memcpy

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef Py_ssize_t i
    cdef float *XX = &X[0]

    for i in range(X.shape[0]):
        X[i] = c

cdef inline void fill_memoryview2(float[:,::1] X, float c):
    cdef Py_ssize_t i
    cdef float *XX = &X[0,0]

    for i in range(X.shape[0] * X.shape[1]):
        XX[i] = c

cdef inline void copy_memoryview(float[::1] Y, float[::1] X):
    cdef Py_ssize_t m = X.shape[0], n = Y.shape[0]

    memcpy(&Y[0], &X[0], (n if n<m else m)*cython.sizeof(float))

cdef class ScalarAverager:
    #
    cdef init(self)
    #
    cdef float update(self, const float x)
    
@cython.final
cdef class ScalarAdaM2(ScalarAverager):
    #
    cdef float m, v, beta1, beta2, beta1_k, beta2_k, epsilon
    #

@cython.final
cdef class ScalarAdaM1(ScalarAverager):
    #
    cdef float m, v, beta1, beta2, beta1_k, beta2_k, epsilon
    #
    
@cython.final
cdef class ScalarExponentialAverager(ScalarAverager):
    #
    cdef float m, beta, beta_k
    #

@cython.final    
cdef class ScalarWindowAverager(ScalarAverager):
    cdef Py_ssize_t size
    cdef Py_ssize_t idx
    cdef float[::1] buffer
    cdef float buffer_sum
    cdef bint first

###########################################################

cdef class ArrayAverager:
    #
    cdef float[::1] array_average
    #
    cdef init(self, object ndim)
    #
    cdef void set_param1(self, float val)
    #
    cdef void set_param2(self, float val)
    #
    cdef void update(self, float[::1] x, float h)

cdef class ArraySave(ArrayAverager):
    pass
    
@cython.final
cdef class ArrayMOM(ArrayAverager):
    #
    cdef float[::1] mgrad
    cdef float beta, M
    cdef bint normalize
    #

# @cython.final
# cdef class ArrayRUD(ArrayAverager):
#     #
#     cdef float[::1] mgrad
#     cdef float beta, M
#     cdef bint normalize
#     #
    
@cython.final
cdef class ArrayAMOM(ArrayAverager):
    #
    cdef float[::1] mgrad
    cdef float beta, M
    cdef bint normalize
    #
    
# @cython.final
# cdef class ArrayUnbiasedMOM(ArrayAverager):
#     #
#     cdef float[::1] mgrad
#     cdef float beta, beta_k
#     cdef bint normalize
#     #
    
@cython.final
cdef class ArrayRMSProp(ArrayAverager):
    #
    cdef float[::1] vgrad
    cdef float beta, M, epsilon
    #
    
@cython.final
cdef class ArrayAdaM2(ArrayAverager):
    #
    cdef float[::1] mgrad, vgrad
    cdef float beta1, beta2, beta1_k, beta2_k, epsilon
    #

@cython.final
cdef class ArrayAdaM1(ArrayAverager):
    #
    cdef float[::1] mgrad, vgrad
    cdef float beta1, beta2, beta1_k, beta2_k, epsilon
    #

# @cython.final
# cdef class ArrayAdaNG(ArrayAverager):
#     #
#     cdef float[::1] mgrad
#     cdef float beta, beta_k, epsilon
#     #

@cython.final
cdef class ArraySimpleAverager(ArrayAverager):
    #
    cdef float[::1] array_sum
    cdef float T
    #

@cython.final
cdef class ArrayCyclicAverager(ArrayAverager):
    #
    cdef Py_ssize_t i, size
    cdef float[:, ::1] array_all
    #

@cython.final
cdef class ArrayTAverager(ArrayAverager):
    #
    cdef float[::1] array_sum
    cdef float T
    #

# @cython.final
# cdef class ArrayExponentialAverager(ArrayAverager):
#     #
#     cdef float beta, beta_k
#     cdef float[::1] array_sum
#     #
