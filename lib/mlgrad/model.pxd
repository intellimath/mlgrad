# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

from mlgrad.func cimport Func

from cpython.object cimport PyObject

cdef extern from *:
    """
    #define D_PTR(p) (*(p))
    """
    double D_PTR(double* p) nogil
    PyObject* PyList_GET_ITEM(PyObject* list, Py_ssize_t i) nogil
    int PyList_GET_SIZE(PyObject* list) nogil
 
cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()

ctypedef double[::1] double_array

ctypedef double (*ModelEvaluate)(Model, double[::1])
ctypedef void (*ModelGradient)(Model, double[::1], double[::1])

ctypedef double (*FuncEvaluate)(Func, double) nogil
ctypedef double (*FuncDerivative)(Func, double) nogil
ctypedef double (*FuncDerivative2)(Func, double) nogil
ctypedef double (*FuncDerivativeDivX)(Func, double) nogil


cdef class Allocator:
    #
    cpdef double[::1] allocate(self, int n)
    cpdef double[:,::1] allocate2(self, int n, int m)
    cpdef double[::1] get_allocated(self)
    cpdef Allocator suballocator(self)

@cython.final
cdef class ArrayAllocator(Allocator):
    cdef ArrayAllocator base
    cdef public int size, start, allocated
    cdef public object buf

cdef inline Model as_model(object o):
    return <Model>(<PyObject*>o)

cdef inline void matrix_dot(double[:,::1] A, double[::1]x, double[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef double v
    
    for j in range(n):
        v = 0
        for i in range(m):
            v += A[j,i] * x[i]
        y[j] = v

cdef inline void matrix_dot_t(double[:,::1] A, double[::1]x, double[::1] y):
    cdef int i, n=A.shape[0], m=A.shape[1]
    cdef double v
    
    for i in range(m):
        v = 0
        for j in range(n):
            v += A[j,i] * x[j]
        y[i] = v

cdef class Model(object):
    cdef public Py_ssize_t n_param, n_input
    cdef public double[::1] param
    cdef public double[::1] grad
    cdef public double[::1] grad_x

    #cdef void init_param(self, double[::1] param=*, bint random=*)
    cdef double evaluate(self, double[::1] X)
    cdef void gradient(self, double[::1] X, double[::1] grad)
    cdef void gradient_x(self, double[::1] X, double[::1] grad)
    cpdef Model copy(self, bint share=*)
    #

@cython.final
cdef class ConstModel(Model):
    pass
    
@cython.final
cdef class LinearModel(Model):
    pass

@cython.final
cdef class SigmaNeuronModel(Model):
    cdef Func outfunc

@cython.final
cdef class WinnerModel(Model):
    cdef Func outfunc

@cython.final
cdef class PolynomialModel(Model):
    pass

cdef class ModelLayer:
    cdef public Py_ssize_t n_param, n_input, n_output
    cdef public double[::1] param
    cdef public double[::1] output
    cdef public double[::1] grad_input
    
    cdef void forward(self, double[::1] X)
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad)
    cpdef ModelLayer copy(self, bint share=*)
    
cdef class SigmaNeuronModelLayer(ModelLayer):
    cdef public Func func
    cdef public double[:,::1] matrix
#     cdef double[::1] ss
#     cdef double[:,::1] X
    
cdef class GeneralModelLayer(ModelLayer):
    cdef public list models

cdef class ComplexModel(object):
    cdef public Py_ssize_t n_param
    cdef public Py_ssize_t n_input, n_output
    cdef public double[::1] param
    cdef public double[::1] output
    
    cdef void forward(self, double[::1] X)
    cdef void backward(self, double[::1] X, double[::1] grad_u, double[::1] grad)
    
cdef class MLModel(ComplexModel):
    cdef public list layers

#     cdef void forward(self, double[::1] X)
#     cdef void backward(self, double[::1] X, double[::1] grad_u, double[::1] grad)
    cpdef MLModel copy(self, bint share=*)

@cython.final
cdef class FFNetworkModel(MLModel):
    pass

@cython.final
cdef class FFNetworkFuncModel(Model):
    #cdef ArrayAllocator allocator_param, allocator_grad
    cdef public Model head
    cdef public MLModel body

@cython.final
cdef class SquaredModel(Model):
    cdef double[:,::1] matrix
    cdef double[:,::1] matrix_grad
    

cdef class MultiModel:
    cdef Model[::1] models
    cdef Py_ssize_t n_model
    cdef double[::1] vals
