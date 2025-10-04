# cython: language_level=3

cimport cython

from mlgrad.models cimport Model, Model2, BaseModel, MLModel
from mlgrad.funcs cimport Func
from mlgrad.funcs2 cimport Func2
from mlgrad.distance cimport Distance
from mlgrad.loss cimport Loss, MultLoss, MultLoss2
from mlgrad.batch cimport Batch, WholeBatch, RandomBatch
#from mlgrad.averager cimport ArrayAverager
from mlgrad.avragg cimport Average, ArithMean
cimport mlgrad.inventory as inventory

# from mlgrad.inventory cimport init_rand, rand, fill

from libc.math cimport fabs, pow, sqrt, fmax
from libc.string cimport memcpy, memset

from cpython.object cimport PyObject
from numpy cimport npy_uint8 as uint8

cdef extern from "Python.h":
    double PyFloat_GetMax()

cdef extern from *:
    PyObject* PyList_GET_ITEM(PyObject* list, Py_ssize_t i) nogil
    int PyList_GET_SIZE(PyObject* list) nogil

# ctypedef double (*ModelEvaluate)(Model, double[::1])
# ctypedef void (*ModelGradient)(Model, double[::1], double[::1])
# ctypedef double (*LossEvaluate)(Loss, double, double)
# ctypedef double (*LossDerivative)(Loss, double, double)

# cdef inline void clear_memoryview(double[::1] X):
#     # cdef int m = X.shape[0]
#     memset(&X[0], 0, X.shape[0]*cython.sizeof(double))

# cdef inline void fill_memoryview(double[::1] X, double c):
#     cdef Py_ssize_t i, m = X.shape[0]
#     for i in range(m):
#         X[i] = c

# cdef inline void clear_memoryview2(double[:, ::1] X):
#     memset(&X[0,0], 0, X.shape[0]*X.shape[1]*cython.sizeof(double))    
        
# cdef inline void fill_memoryview2(double[:,::1] X, double c):
#     cdef int i, j
#     cdef int m = X.shape[0], n = X.shape[1]
#     for i in range(m):
#         for j in range(n):
#             X[i,j] = c

# cdef inline void copy_memoryview(double[::1] Y, double[::1] X):
#     cdef int m = X.shape[0], n = Y.shape[0]

#     if n < m:
#         m = n
#     memcpy(&Y[0], &X[0], m*cython.sizeof(double))    

cdef class Functional:
    cdef readonly double[::1] param
    cdef readonly double[::1] grad_average
    cdef readonly double lval
    cdef readonly Batch batch
    cdef readonly Py_ssize_t n_param
    cdef readonly Py_ssize_t n_input
    cdef readonly bint is_natgrad

    cpdef init(self)
    cdef public double _evaluate(self)
    cdef public void _gradient(self)
    cdef void _update_param(self, double[::1] param)

cdef class SimpleFunctional(Functional):
    cdef readonly Func2 regnorm

cdef class Risk(Functional):
    #
    cdef readonly double[::1] weights
    cdef readonly double[::1] Yp
    cdef readonly double[::1] L
    # cdef readonly double[::1] LD
    #
    cdef double[::1] grad
    cdef double[::1] grad_r
    # cdef readonly double tau
    cdef readonly Py_ssize_t n_sample
    cdef uint8[::1] mask
    #
    # cdef void _evaluate_models_all(self, double[::1] vals)
    # cdef void _evaluate_models_batch(self)
    cdef void _evaluate_losses_batch(self)
    cdef void _evaluate_losses_all(self, double[::1] lvals)
    # cdef void _evaluate_losses_derivative_div_batch(self)
    cdef void _evaluate_losses_derivative_div_all(self, double[::1] vals)
    cdef void _evaluate_weights(self)
    cdef void add_regularized_gradient(self)
    #
    #
    # cdef void generate_samples(self, X, Y)z
    #

cdef class ERisk(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y

cdef class ERiskGB(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
    cdef readonly double[::1] H
    cdef public double alpha
    
    cdef double derivative_alpha(self)
    
cdef class MRisk(Risk):
    cdef readonly Model model
    cdef readonly Loss loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
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
    

cdef class ERisk22(Risk):
    cdef readonly MLModel model
    cdef readonly MultLoss2 loss
    cdef readonly double[:, ::1] X
    cdef readonly double[:, ::1] Y
    cdef double[::1] grad_u
    cdef readonly Py_ssize_t n_output

cdef class ERisk2(Risk):
    cdef readonly Model2 model
    cdef readonly MultLoss2 loss
    cdef readonly double[:, ::1] X
    cdef readonly double[::1] Y
    cdef double[::1] grad_u
    cdef readonly Py_ssize_t n_output
