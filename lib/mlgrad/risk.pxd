# cython: language_level=3

cimport cython

from mlgrad.model cimport Model, MLModel
from mlgrad.func cimport Func
from mlgrad.distance cimport Distance
from mlgrad.loss cimport Loss, MultLoss
from mlgrad.regular cimport FuncMulti
from mlgrad.batch cimport Batch, WholeBatch
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
    
cdef inline void fill_memoryview(double[::1] X, double c):
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

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
    cdef public FuncMulti regular
    cdef public double[::1] param
    cdef public double[::1] grad_average
    cdef public double lval
    cdef public Batch batch
    cdef public int n_sample

    cpdef init(self)
    cdef double evaluate(self)
    cdef void gradient(self)

cdef class SimpleFunctional(Functional):
    pass

cdef class Risk(Functional):
    #
    cdef double[::1] grad
    cdef double[::1] grad_r
    cdef public double[::1] weights    
    cdef public double tau
    #
    cdef double eval_loss(self, int k)
    cdef void gradient_loss(self, int k)
    cdef void eval_losses(self, double[::1] lval_all)
    #cdef object get_loss(self)

cdef class ED(Risk):
    cdef double[:, ::1] X
    cdef Distance distfunc
    cdef int n_param
    #

cdef class ER(Risk):
    cdef public Model model
    cdef public Loss loss
    cdef public double[:, ::1] X
    cdef public double[::1] Y
#     cdef list models
#     cdef double[:,::1] grads

cdef class AER(ER):
    cdef Average loss_averager
    cdef double[::1] lval_all
    cdef double[::1] mval_all
    
    cdef eval_all(self)
    

cdef class ER2(Risk):
    cdef public MLModel model
    cdef public MultLoss loss
    cdef double[:, ::1] X
    cdef double[:, ::1] Y
    cdef double[::1] grad_u
