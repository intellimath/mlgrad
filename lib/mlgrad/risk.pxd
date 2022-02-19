# cython: language_level=3

cimport cython

from mlgrad.model cimport Model, MLModel, BaseModel
from mlgrad.func cimport Func
from mlgrad.distance cimport Distance
from mlgrad.loss cimport Loss, MultLoss, MultLoss2
from mlgrad.regnorm cimport FuncMulti
from mlgrad.batch cimport Batch, WholeBatch, RandomBatch
#from mlgrad.averager cimport ArrayAverager
from mlgrad.avragg cimport Average, ArithMean
#from mlgrad.weights cimport Weights

from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

from cpython.object cimport PyObject

cdef extern from "Python.h":
    double PyFloat_GetMax()

cdef extern from *:
    PyObject* PyList_GET_ITEM(PyObject* list, Py_ssize_t i) nogil
    int PyList_GET_SIZE(PyObject* list) nogil

ctypedef double (*ModelEvaluate)(Model, double[::1])
ctypedef void (*ModelGradient)(Model, double[::1], double[::1])
ctypedef double (*LossEvaluate)(Loss, double, double)
ctypedef double (*LossDerivative)(Loss, double, double)

cdef inline void clear_memoryview(double[::1] X):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void fill_memoryview(double[::1] X, double c):
    cdef Py_ssize_t i, m = X.shape[0]
    for i in range(m):
        X[i] = c

cdef inline void clear_memoryview2(double[:, ::1] X):
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(double))    
        
cdef inline void fill_memoryview2(double[:,::1] X, double c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    for i in range(m):
        for j in range(n):
            X[i,j] = c

cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))    

cdef class Functional:
    cdef readonly FuncMulti regnorm
    cdef readonly double[::1] param
    cdef readonly double[::1] grad_average
    cdef readonly double lval
    cdef readonly Batch batch
    cdef readonly Py_ssize_t n_param
    cdef readonly Py_ssize_t n_input

    cpdef init(self)
    cdef public double _evaluate(self)
    cdef public void _gradient(self)

cdef class SimpleFunctional(Functional):
    pass

cdef class Risk(Functional):
    #
    cdef readonly double[::1] weights
    cdef readonly double[::1] Yp
    cdef readonly double[::1] L
    cdef readonly double[::1] LD
    #
    cdef double[::1] grad
    cdef double[::1] grad_r
    # cdef public double[::1] H
    cdef readonly double tau
    cdef readonly Py_ssize_t n_sample
    #
    cdef void _evaluate_models(self)
    cdef void _evaluate_losses(self)
    cdef void _evaluate_losses_derivative_div(self)
    cdef void _evaluate_weights(self)
    #
    
# cdef class SRisk(Risk):
#     cdef public double eval_loss(self, int k)
#     cdef public void gradient_loss(self, int k)    

cdef class ERisk(Risk):
    cdef public Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y

cdef class ERiskGB(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
    # cdef readonly double[::1] L
    # cdef readonly double[::1] LD
    cdef readonly double[::1] H
    cdef public double alpha
    
    cdef double derivative_alpha(self)
    
cdef class MRisk(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
    # cdef readonly double[::1] Yp
    # cdef readonly double[::1] L
    # cdef readonly double[::1] LD
    # cdef readonly double[::1] lval_all
    cdef Average avg
    cdef bint first
    
cdef class ED(Risk):
    cdef readonly double[:, ::1] X
    cdef readonly Distance distfunc
    #
    
# cdef class AER(ERisk):
#     cdef Average loss_averager
#     cdef double[::1] lval_all
#     cdef double[::1] mval_all
    
#     cdef eval_all(self)
    

cdef class ER2(Risk):
    cdef readonly MLModel model
    cdef readonly MultLoss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[:, ::1] Y
    cdef double[::1] grad_u
    cdef readonly Py_ssize_t n_output

cdef class ER21(Risk):
    cdef readonly MLModel model
    cdef readonly MultLoss2 loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
    cdef double[::1] grad_u
    cdef readonly Py_ssize_t n_output
