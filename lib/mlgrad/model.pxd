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
    double Pydouble_GetMax()
    double Pydouble_GetMin()

ctypedef double[::1] double_array

ctypedef double (*ModelEvaluate)(Model, double[::1])
ctypedef void (*ModelGradient)(Model, double[::1], double[::1])

ctypedef double (*FuncEvaluate)(Func, double) nogil
ctypedef double (*FuncDerivative)(Func, double) nogil
ctypedef double (*FuncDerivative2)(Func, double) nogil
ctypedef double (*FuncDerivativeDivX)(Func, double) nogil

from mlgrad.list_values cimport list_doubles

cimport mlgrad.inventory as inventory

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

cdef class BaseModel(object):
    cdef double _evaluate(self, double[::1] X)
    cdef _evaluate_all(self, double[:,::1] X, double[::1] Y)
    cpdef copy(self, bint share=*)

cdef class Model(BaseModel):
    cdef public Py_ssize_t n_param, n_input
    cdef public double[::1] param
    cdef public double[::1] grad
    cdef public double[::1] grad_x

    cpdef init_param(self, param=*, bint random=*)
    # cdef double evaluate(self, double[::1] X)
    cdef void _gradient(self, double[::1] X, double[::1] grad)
    cdef void _gradient_x(self, double[::1] X, double[::1] grad)
    #

# @cython.final
# cdef class ConstModel(Model):
#     pass
    
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

# cdef class ModelAdd(Model):
#     cdef Model main
#     cdef Model base
#     cdef double alpha

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
    cdef double[::1] ss
    cdef bint first_time
    
cdef class GeneralModelLayer(ModelLayer):
    cdef public list models

# cdef class ComplexModel(object):
#     cdef double evaluate(self, double[::1] X)
#     cpdef copy(self, bint share=*)
    
cdef class LinearFuncModel(BaseModel):
    cdef public list models
    cdef public list_doubles weights
    
    
cdef class MLModel:
    cdef public Py_ssize_t n_param
    cdef public Py_ssize_t n_input, n_output
    cdef public double[::1] param
    cdef public double[::1] output
    cdef public list layers
    cdef bint is_forward

    cdef void forward(self, double[::1] X)
    cdef void backward(self, double[::1] X, double[::1] grad_u, double[::1] grad)    
    cdef void backward2(self, double[::1] X, double[::1] grad_u, double[::1] grad)
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
                                                                    