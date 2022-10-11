#
# Least Squares for Simple Linear Models
#

from mlgrad cimport LinearModel


cdef void linear_ls(double[::1] X, double[::1] Y, LinearModel mod)
cdef double linear_ls_slope(double[::1] X, double[::1] Y)
cdef double linear_ls_slope2(double[::1] Y)

cdef void linear_wls(double[::1] X, double[::1] Y, double[::1] W, LinearModel mod)
cdef double linear_wls_slope(double[::1] X, double[::1] Y, double[::1] W)
cdef double linear_wls_slope2(double[::1] Y, double[::1] W)
