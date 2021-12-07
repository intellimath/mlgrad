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
    float D_PTR(float* p) nogil
    PyObject* PyList_GET_ITEM(PyObject* list, Py_ssize_t i) nogil
    int PyList_GET_SIZE(PyObject* list) nogil
 
cdef extern from "Python.h":
    float PyFloat_GetMax()
    float PyFloat_GetMin()

ctypedef float[::1] float_array

ctypedef float (*ModelEvaluate)(Model, float[::1])
ctypedef void (*ModelGradient)(Model, float[::1], float[::1])

ctypedef float (*FuncEvaluate)(Func, float) nogil
ctypedef float (*FuncDerivative)(Func, float) nogil
ctypedef float (*FuncDerivative2)(Func, float) nogil
ctypedef float (*FuncDerivativeDivX)(Func, float) nogil

cdef extern from "c/inventory.h" nogil:
    float dconv(const float*, const float*, const int)
    void dmove(float*, const float*, const int)
    float dsum(const float*, const int)
    void dfill(float*, const float, const int)
    void dmatdot(float*, float*, const float*, const size_t, const size_t)
    void dmatdot2(float*, float*, const float*, const size_t, const size_t)
    void dmult_add_arrays(float *a, const float *b, const float *ss, const size_t n_input, const size_t n_output)
    void dmult_grad(float *grad, const float *X, const float *ss, const size_t n_input, const size_t n_output)

from mlgrad.list_values cimport list_values

cimport mlgrad.inventory as inventory

cdef class Allocator:
    #
    cpdef float[::1] allocate(self, int n)
    cpdef float[:,::1] allocate2(self, int n, int m)
    cpdef float[::1] get_allocated(self)
    cpdef Allocator suballocator(self)

@cython.final
cdef class ArrayAllocator(Allocator):
    cdef ArrayAllocator base
    cdef public int size, start, allocated
    cdef public object buf

cdef inline Model as_model(object o):
    return <Model>(<PyObject*>o)

cdef class Model(object):
    cdef public Py_ssize_t n_param, n_input
    cdef public float[::1] param
    cdef public float[::1] grad
    cdef public float[::1] grad_x

    cpdef init_param(self, param=*, bint random=*)
    cdef float evaluate(self, float[::1] X)
    cdef void gradient(self, float[::1] X, float[::1] grad)
    cdef void gradient_x(self, float[::1] X, float[::1] grad)
    cpdef Model copy(self, bint share=*)
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

cdef class ModelAdd(Model):
    cdef Model main
    cdef Model base
    cdef float alpha

cdef class ModelLayer:
    cdef public Py_ssize_t n_param, n_input, n_output
    cdef public float[::1] param
    cdef public float[::1] output
    cdef public float[::1] grad_input
    
    cdef void forward(self, float[::1] X)
    cdef void backward(self, float[::1] X, float[::1] grad_out, float[::1] grad)
    cpdef ModelLayer copy(self, bint share=*)
    
cdef class SigmaNeuronModelLayer(ModelLayer):
    cdef public Func func
    cdef public float[:,::1] matrix
    cdef float[::1] ss
    cdef bint first_time
    
cdef class GeneralModelLayer(ModelLayer):
    cdef public list models

cdef class ComplexModel(object):
    cdef public Py_ssize_t n_param
    cdef public Py_ssize_t n_input, n_output
    cdef public float[::1] param
    cdef public float[::1] output
    
    cdef void forward(self, float[::1] X)
    cdef void backward(self, float[::1] X, float[::1] grad_u, float[::1] grad)
    
cdef class MLModel(ComplexModel):
    cdef public list layers

#     cdef void forward(self, float[::1] X)
#     cdef void backward(self, float[::1] X, float[::1] grad_u, float[::1] grad)
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
    cdef float[:,::1] matrix
    cdef float[:,::1] matrix_grad
         

cdef class MultiModel:
    cdef Model[::1] models
    cdef Py_ssize_t n_model
    cdef float[::1] vals
                                                                    