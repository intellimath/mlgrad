#

cimport mlgrad.inventory as inventory
from libc.math cimport isnan, fma, sqrt, fabs, log

cdef class DistrFunc:
    cdef public double loc, scale
    #
    cdef double _evaluate_probability(double x)
    #
    cdef void _evaluate_probability_array(double[::1] xs, double[::1] ys)
    #
    cdef double _evaluate_density(double x)
    #
    cdef void _evaluate_density_array(double[::1] xs, double[::1] ys)
    #
    cdef double _evaluate_loss(double x)
    #
    cdef void _evaluate_loss_array(double[::1] xs, double[::1] ys)
    #
    cdef double _derivative_loss(double x)
    #
    cdef void _derivative_loss_array(double[::1] xs, double[::1] ys)
    #
    cdef double _derivative_loss_loc(double x)
    #
    cdef void _derivative_loss_loc_array(double[::1] xs, double[::1] ys)
    #
    cdef double _derivative_loss_scale(double x)
    #
    cdef void _derivative_loss_scale_array(double[::1] xs, double[::1] ys)
    #
    cdef double _scale_next(self, double[::1] X)
    #
    cdef double _loc_next(self, double[::1] X)

cdef class SigmoidalDistrFunc(DistrFunc):
    #
    pass
    
