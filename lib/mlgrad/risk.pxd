# cython: language_level=3

cimport cython

from mlgrad.model cimport Model, MLModel
from mlgrad.func cimport Func
from mlgrad.distance cimport Distance
from mlgrad.loss cimport Loss, MultLoss, MultLoss2
from mlgrad.regnorm cimport FuncMulti
from mlgrad.batch cimport Batch, WholeBatch
#from mlgrad.averager cimport ArrayAverager
from mlgrad.avragg cimport Average, ArithMean
#from mlgrad.weights cimport Weights

from mlgrad.miscfuncs cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

from cpython.object cimport PyObject

cdef extern from "Python.h":
    float PyFloat_GetMax()

cdef extern from *:
    PyObject* PyList_GET_ITEM(PyObject* list, Py_ssize_t i) nogil
    int PyList_GET_SIZE(PyObject* list) nogil

ctypedef float (*ModelEvaluate)(Model, float[::1])
ctypedef void (*ModelGradient)(Model, float[::1], float[::1])
ctypedef float (*LossEvaluate)(Loss, float, float)
ctypedef float (*LossDerivative)(Loss, float, float)

cdef inline void clear_memoryview(float[::1] X):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(float))    

cdef inline void fill_memoryview(float[::1] X, float c):
    cdef Py_ssize_t i, m = X.shape[0]
    for i in range(m):
        X[i] = c

cdef inline void clear_memoryview2(float[:, ::1] X):
    cdef int m = X.shape[0], n = X.shape[1]
    memset(&X[0,0], 0, m*n*cython.sizeof(float))    
        
cdef inline void fill_memoryview2(float[:,::1] X, float c):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    for i in range(m):
        for j in range(n):
            X[i,j] = c

cdef inline void copy_memoryview(float[::1] Y, float[::1] X):
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(float))    

cdef class Functional:
    cdef readonly FuncMulti regnorm
    cdef readonly float[::1] param
    cdef readonly float[::1] grad_average
    cdef readonly float lval
    cdef readonly Batch batch
    cdef readonly Py_ssize_t n_sample
    cdef readonly Py_ssize_t n_param
    cdef readonly Py_ssize_t n_input

    cpdef init(self)
    cdef public float evaluate(self)
    cdef public void gradient(self)

cdef class SimpleFunctional(Functional):
    pass

cdef class Risk(Functional):
    #
    cdef float[::1] grad
    cdef float[::1] grad_r
    cdef readonly float[::1] weights    
    cdef readonly float tau
    #
    cdef public void eval_losses(self, float[::1] lval_all)
    
# cdef class SRisk(Risk):
#     cdef public float eval_loss(self, int k)
#     cdef public void gradient_loss(self, int k)    

cdef class ERisk(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly float[:, ::1] X
    cdef readonly float[::1] Y
    cdef readonly float[::1] Yp
    
cdef class MRisk(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly float[:, ::1] X
    cdef readonly float[::1] Y
    cdef readonly float[::1] Yp
    cdef readonly float[::1] lval_all
    cdef Average avg
    cdef bint first
    
cdef class ED(Risk):
    cdef readonly float[:, ::1] X
    cdef readonly Distance distfunc
    #
    
cdef class AER(ERisk):
    cdef Average loss_averager
    cdef float[::1] lval_all
    cdef float[::1] mval_all
    
    cdef eval_all(self)
    

cdef class ER2(Risk):
    cdef readonly MLModel model
    cdef readonly MultLoss loss
    cdef readonly float[:, ::1] X
    cdef readonly float[:, ::1] Y
    cdef float[::1] grad_u
    cdef readonly Py_ssize_t n_output

cdef class ER21(Risk):
    cdef readonly MLModel model
    cdef readonly MultLoss2 loss
    cdef readonly float[:, ::1] X
    cdef readonly float[::1] Y
    cdef float[::1] grad_u
    cdef readonly Py_ssize_t n_output
