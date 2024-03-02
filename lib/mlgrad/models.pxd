# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False

cimport cython

from libc.math cimport fabs, pow, sqrt, fmax, log, exp
from libc.string cimport memcpy, memset

from mlgrad.funcs cimport Func
from mlgrad.funcs2 cimport Func2

from cpython.object cimport PyObject

cdef extern from *:
    """
    #define D_PTR(p) (*(p))
    """
    double D_PTR(double* p) noexcept nogil
    PyObject* PyList_GET_ITEM(list ob, Py_ssize_t i) noexcept nogil
    int PyList_GET_SIZE(list ob) noexcept nogil
 
cdef extern from "Python.h":
    double PyDouble_GetMax()
    double PyDouble_GetMin()

ctypedef double[::1] double_array
ctypedef Model[::1] ModelArray

ctypedef double (*ModelEvaluate)(Model, double[::1])
ctypedef void (*ModelGradient)(Model, double[::1], double[::1])

ctypedef double (*ptrfunc)(double) nogil
# ctypedef double (*FuncDerivative)(Func, double) nogil
# ctypedef double (*FuncDerivative2)(Func, double) nogil
# ctypedef double (*FuncDerivativeDivX)(Func, double) nogil

from mlgrad.list_values cimport list_doubles
from mlgrad.array_allocator cimport Allocator, ArrayAllocator

cimport mlgrad.inventory as inventory

cdef inline Model as_model(object o):
    return <Model>(<PyObject*>o)

cdef class BaseModel(object):
    cdef double _evaluate(self, double[::1] X)
    cdef void _evaluate_all(self, double[:,::1] X, double[::1] Y)
    cpdef copy(self, bint share=*)

cdef class Model(BaseModel):
    cdef public Py_ssize_t n_param, n_input
    cdef public object ob_param
    cdef public double[::1] param
    cdef public double[::1] grad
    cdef public double[::1] grad_input
    # cdef bint is_allocated

    # cpdef init_param(self, param=*, bint random=*)
    cdef void _gradient(self, double[::1] X, double[::1] grad)
    cdef void _gradient_input(self, double[::1] X, double[::1] grad)
    #
    # cdef update_param(self, double[::1] param)

cdef class SimpleComposition(Model):
    #
    cdef Func func
    cdef Model model
    
cdef class ModelComposition(Model):
    #
    # cdef Py_ssize_t n_input
    cdef Func2 func
    cdef public list models
    cdef double[::1] ss, sx
    
    cdef void _gradient_j(self, double[::1] X, Py_ssize_t j, double[::1] grad)
    
cdef class ModelComposition_j(Model):
    cdef ModelComposition model_comp
    cdef Model model_j
    cdef Py_ssize_t j
    
    
cdef class ModelView(Model):
    cdef Model model
    
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

cdef class Model2:
    cdef public Py_ssize_t n_param, n_input, n_output
    cdef public object ob_param
    cdef public double[::1] param
    cdef public double[::1] output
    #
    cdef void _forward(self, double[::1] X)
    #
    cdef void gradient_j(self, Py_ssize_t j, double[::1] X, double[::1] grad)
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad)
    #

cdef class ModelLayer(Model2):
    cdef public double[::1] input
    cdef public double[::1] grad_input
    
    cpdef ModelLayer copy(self, bint share=*)
    
@cython.final
cdef class ScaleLayer(ModelLayer):
    cdef public Func func    
    
@cython.final
cdef class LinearLayer(ModelLayer):
    cdef public double[:,::1] matrix
    # cdef bint first_time

cdef class SigmaNeuronModelLayer(ModelLayer):
    cdef public Func func
    cdef public double[:,::1] matrix
    cdef double[::1] ss
    cdef bint first_time
    
cdef class GeneralModelLayer(ModelLayer):
    cdef public list models
    
cdef class LinearFuncModel(BaseModel):
    cdef public list models
    cdef public list_doubles weights
    
    
cdef class MLModel(Model2):
    cdef public list layers
    cdef bint is_forward

    cdef void backward2(self, double[::1] X, double[::1] grad_u, double[::1] grad)
    cpdef MLModel copy(self, bint share=*)

# @cython.final
cdef class FFNetworkModel(MLModel):
    pass

@cython.final
cdef class FFNetworkFuncModel(Model):
    #cdef ArrayAllocator allocator_param, allocator_grad
    cdef public Model head
    cdef public MLModel body    
    
cdef class EllipticModel(Model):
    cdef readonly double[::1] c
    cdef readonly double[::1] S
    cdef double[::1] grad_c
    cdef double[::1] grad_S
    cdef Py_ssize_t c_size, S_size

    cdef _gradient_c(self, double[::1] X, double[::1] grad)
    cdef _gradient_S(self, double[::1] X, double[::1] grad)
    
@cython.final
cdef class SquaredModel(Model):
    cdef double[:,::1] matrix
    cdef double[:,::1] matrix_grad
                                                                    
@cython.final
cdef class LogSpecModel(Model):
    cdef public double[::1] center
    cdef public double[::1] scale
