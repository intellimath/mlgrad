# cimport mlgrad.funcs2 as funcs2
# cimport mlgrad.funcs as funcs
# cimport mlgrad.averager as averager
cimport mlgrad.inventory as inventory
# cimport mlgrad.averager as averager

from libc.math cimport fabs, pow, sqrt, fmax, exp, log, atan, fma

# cdef class WhittakerSmoother:
#     cdef readonly funcs2.Func2 func
#     cdef readonly funcs2.Func2 func2
#     cdef public double h, tol, tau, alpha
#     cdef public Py_ssize_t n_iter
#     cdef readonly Py_ssize_t K
#     cdef public double[::1] Z
#     cdef readonly list qvals
#     cdef readonly double qval, delta_qval

