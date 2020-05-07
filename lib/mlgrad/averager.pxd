# cython: language_level=3

cimport cython
from libc.string cimport memcpy

cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int i, m = X.shape[0]
    for i in range(m):
        X[i] = c

cdef inline void fill_memoryview2(double[:,::1] X, double c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    for i in range(m):
        for j in range(n):
            X[i,j] = c

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    cdef int i, m = X.shape[0], n = Y.shape[0]
    #cdef double* X_ptr = &X[0]
    #cdef double* Y_ptr = &Y[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))
    #for i in range(m):
    #    Y_ptr[i] = X_ptr[i]

cdef class ScalarAverager:
    #
    cdef init(self)
    #
    cdef double update(self, double x)
    
@cython.final
cdef class ScalarAdaM2(ScalarAverager):
    #
    cdef double m, v, beta1, beta2, beta1_k, beta2_k, epsilon
    #

@cython.final
cdef class ScalarAdaM1(ScalarAverager):
    #
    cdef double m, v, beta1, beta2, beta1_k, beta2_k, epsilon
    #
    
@cython.final
cdef class ScalarExponentialAverager(ScalarAverager):
    #
    cdef double m, beta, beta_k
    #

@cython.final    
cdef class ScalarWindowAverager(ScalarAverager):
    cdef int size
    cdef int idx
    cdef double[::1] buffer
    cdef double buffer_sum
    cdef bint first

###########################################################

cdef class ArrayAverager:
    #
    cdef double[::1] array_average
    #
    cdef init(self, object ndim)
    #
    cdef void set_param1(self, double val)
    #
    cdef void set_param2(self, double val)
    #
    cdef void update(self, double[::1] x, double h)

cdef class ArraySave(ArrayAverager):
    pass
    
@cython.final
cdef class ArrayMOM(ArrayAverager):
    #
    cdef double[::1] mgrad
    cdef double beta, M
    cdef bint normalize
    #

# @cython.final
# cdef class ArrayRUD(ArrayAverager):
#     #
#     cdef double[::1] mgrad
#     cdef double beta, M
#     cdef bint normalize
#     #
    
@cython.final
cdef class ArrayAMOM(ArrayAverager):
    #
    cdef double[::1] mgrad
    cdef double beta, M
    cdef bint normalize
    #
    
# @cython.final
# cdef class ArrayUnbiasedMOM(ArrayAverager):
#     #
#     cdef double[::1] mgrad
#     cdef double beta, beta_k
#     cdef bint normalize
#     #
    
@cython.final
cdef class ArrayRMSProp(ArrayAverager):
    #
    cdef double[::1] vgrad
    cdef double beta, M, epsilon
    #
    
@cython.final
cdef class ArrayAdaM2(ArrayAverager):
    #
    cdef double[::1] mgrad, vgrad
    cdef double beta1, beta2, beta1_k, beta2_k, epsilon
    #

@cython.final
cdef class ArrayAdaM1(ArrayAverager):
    #
    cdef double[::1] mgrad, vgrad
    cdef double beta1, beta2, beta1_k, beta2_k, epsilon
    #

# @cython.final
# cdef class ArrayAdaNG(ArrayAverager):
#     #
#     cdef double[::1] mgrad
#     cdef double beta, beta_k, epsilon
#     #

@cython.final
cdef class ArraySimpleAverager(ArrayAverager):
    #
    cdef double[::1] array_sum
    cdef double T
    #

@cython.final
cdef class ArrayCyclicAverager(ArrayAverager):
    #
    cdef int i, size
    cdef double[:, ::1] array_all
    #

@cython.final
cdef class ArrayTAverager(ArrayAverager):
    #
    cdef double[::1] array_sum
    cdef double T
    #

# @cython.final
# cdef class ArrayExponentialAverager(ArrayAverager):
#     #
#     cdef double beta, beta_k
#     cdef double[::1] array_sum
#     #
