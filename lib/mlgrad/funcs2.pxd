# cython: language_level=3

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax, exp, log, fma, copysign
from libc.string cimport memcpy, memset

from cython.parallel cimport parallel, prange
cimport mlgrad.inventory as inventory
from mlgrad.funcs cimport Func
from mlgrad.list_values cimport list_int, list_double
cimport numpy

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

cdef double double_max

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

cdef class ProjectToSubspace:
    cdef Py_ssize_t n, m
    cdef public double[::1] w0
    cdef public double[::1] w
    cdef double[::1] dw
    cdef public list eqns
    # cdef list_double lams
    cdef double[:,::1] A
    cdef double[:,::1] G
    cdef double[::1] b
    cdef double tol
    cdef Py_ssize_t n_iter

cdef class Func2:
    #cdef bint all
    cdef void _evaluate_items(self, double[::1] param, double[::1] vals) noexcept nogil
    cdef double _evaluate(self, double[::1] param) noexcept nogil
    cdef void _gradient(self, double[::1] param, double[::1] grad) noexcept nogil
    cdef double _evaluate_ex(self, double[::1] param, double[::1] weights) noexcept nogil
    cdef void _gradient_ex(self, double[::1] param, double[::1] grad, double[::1] weights) noexcept nogil
    cdef double _gradient_j(self, double[::1] X, Py_ssize_t j) noexcept nogil
    cdef void _normalize(self, double[::1] X) noexcept nogil

cdef class Dot(Func2):
    cdef double[::1] a
    cdef Py_ssize_t offset

cdef class FuncDot(Func2):
    cdef Func func
    cdef double[::1] a
    cdef Py_ssize_t offset

# cdef class Func2Layer:
#     cdef void _evaluate(self, double[::1] X, double[::1] Y)
#     cdef void _gradient(self, double[::1] X, double[::1] Y)

# cdef class SquareNormLayer(Func2Layer):
#     cdef list funcs
#     cdef list_int starts
#     cdef list_int counts

@cython.final
cdef class FuncNorm(Func2):
    #
    cdef Func func
    #

cdef class MixedNorm(Func2):
    #
    cdef Func2 func1
    cdef Func2 func2
    cdef double tau1, tau2

@cython.final
cdef class PowerNorm(Func2):
    #
    cdef Py_ssize_t offset
    cdef double p
    #

@cython.final
cdef class SquareNorm(Func2):
    #
    cdef Py_ssize_t offset
    #

@cython.final
cdef class AbsoluteNorm(Func2):
    #
    cdef Py_ssize_t offset
    #

@cython.final
cdef class SoftAbsoluteNorm(Func2):
    cdef double eps, eps2

@cython.final
cdef class SoftPowerAbsoluteNorm(Func2):
    cdef double p

@cython.final
cdef class SquareForm(Func2):
    cdef double[:,::1] matrix

@cython.final
cdef class Rosenbrok(Func2):
    pass

@cython.final
cdef class Himmelblau(Func2):
    pass

@cython.final
cdef class SoftMin(Func2):
    cdef double p
    cdef double[::1] evals

@cython.final
cdef class SoftMax(Func2):
    cdef double p
    cdef double[::1] evals

@cython.final
cdef class PowerMax(Func2):
    cdef double p
    cdef double[::1] evals

@cython.final
cdef class SquareDiff1(Func2):
    pass

# @cython.final
# cdef class SquareDiff2(Func2):
#     pass

@cython.final
cdef class FuncDiff2(Func2):
    cdef readonly Func func
    cdef double[::1] temp_array
    #
    cdef void _evaluate_diff2(self, double *XX, double *YY, const Py_ssize_t m) noexcept nogil

