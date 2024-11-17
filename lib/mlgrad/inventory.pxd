# coding: utf-8 

# cython: language_level=3

cimport cython

from numpy cimport npy_uint8 as uint8

from libc.string cimport memcpy, memset
from libc.math cimport isnan, fma, sqrt, fabs
from libc.stdlib cimport rand as stdlib_rand, srand
from libc.time cimport time

cdef extern from "pymath.h" nogil:
    bint Py_IS_FINITE(double x)
    bint Py_IS_INFINITY(double x)
    bint Py_IS_NAN(double x)
    bint copysign(double x, double x)

cdef extern from "Python.h":
    double PyFloat_GetMax()
    double PyFloat_GetMin()   

cdef void init_rand() noexcept nogil
cdef long rand(long N) noexcept nogil

cdef int get_num_threads() noexcept nogil
cdef int get_num_procs() noexcept nogil
cdef int get_num_threads_ex(int n) noexcept nogil
cdef int get_num_procs_ex(int n) noexcept nogil
# cdef void set_num_threads(int num) noexcept nogil

cdef int _hasnan(double *a, const Py_ssize_t n) noexcept nogil

cdef double _min(double *a, Py_ssize_t n) noexcept nogil

cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil
cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil
cdef double _conv(const double*, const double*, const Py_ssize_t) noexcept nogil
cdef void _move(double*, const double*, const Py_ssize_t) noexcept nogil
cdef void _move_t(double *to, const double *src, const Py_ssize_t n, const Py_ssize_t step) noexcept nogil
cdef double _sum(const double*, const Py_ssize_t) noexcept nogil

cdef void _iadd(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _add(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil

cdef void _isub(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _isub_mask(double *a, const double *b, uint8 *m, const Py_ssize_t n) noexcept nogil
cdef void _sub(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil

cdef void _imul(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _mul(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil

cdef void _mul_add(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_set(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_set1(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_const(double *a, const double c, const Py_ssize_t n) noexcept nogil
cdef double _dot1(const double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef double _dot(const double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef double _dot_t(const double *a, double *b, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void _matdot(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) noexcept nogil
cdef void _matdot2(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) noexcept nogil
cdef void _mul_add_arrays(double *a, double *M, const double *ss, 
                          const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil
cdef void _mul_grad(double *grad, const double *X, const double *ss, 
                    const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil
cdef void _normalize(double *a, const Py_ssize_t n) noexcept nogil
cdef void _normalize2(double *a, const Py_ssize_t n) noexcept nogil

cdef int hasnan(double[::1] a) noexcept nogil

cdef void clear(double[::1] to) noexcept nogil
cdef void clear2(double[:,::1] to) noexcept nogil
cdef void fill(double[::1] to, const double c) noexcept nogil
cdef void move(double[::1] to, double[::1] src) noexcept nogil
cdef void move2(double[:,::1] to, double[:,::1] src) noexcept nogil
cdef void move3(double[:,:,::1] to, double[:,:,::1] src) noexcept nogil
cdef double conv(double[::1] a, double[::1] b) noexcept nogil
cdef double sum(double[::1] a) noexcept nogil
cdef void iadd(double[::1] a, double[::1] b) noexcept nogil
cdef void iadd2(double[:,::1] a, double[:,::1] b) noexcept nogil
cdef void add(double[::1] c, double[::1] a, double[::1] b) noexcept nogil
cdef void isub(double[::1] a, double[::1] b) noexcept nogil
cdef void isub_mask(double[::1] a, double[::1] b, uint8[::1] m) noexcept nogil
cdef void sub(double[::1] c, double[::1] a, double[::1] b) noexcept nogil
cdef void mul_const(double[::1] a, const double c) noexcept nogil
cdef void mul_const2(double[:, ::1] a, const double c) noexcept nogil
cdef void mul_const3(double[:,:,::1] a, const double c) noexcept nogil
cdef void imul(double[::1] a, double[::1] b) noexcept nogil
cdef void imul2(double[:,::1] a, double[:,::1] b) noexcept nogil
cdef void mul(double[::1] c, double[::1] a, double[::1] b) noexcept nogil
cdef void mul_add(double[::1] a, double[::1] b, const double c) noexcept nogil
cdef void mul_add2(double[:,::1] a, double[:,::1] b, const double c) noexcept nogil
cdef void mul_set(double[::1] a, double[::1] b, const double c) noexcept nogil
cdef void mul_set1(double[::1] a, double[::1] b, const double c) noexcept nogil
cdef double dot1(double[::1] a, double[::1] b) noexcept nogil
cdef double dot(double[::1] a, double[::1] b) noexcept nogil
cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil
cdef void matdot2(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil
cdef void mul_add_arrays(double[::1] a, double[:,::1] M, double[::1] ss) noexcept nogil
cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) noexcept nogil
cdef void normalize(double[::1] a) noexcept nogil
cdef void normalize2(double[::1] a) noexcept nogil

cdef void scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil
cdef void scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil
cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) noexcept nogil

cdef object empty_array(Py_ssize_t size)
cdef object empty_array2(Py_ssize_t size1, Py_ssize_t size2)
cdef object zeros_array(Py_ssize_t size)
cdef object zeros_array2(Py_ssize_t size1, Py_ssize_t size2)

cdef object diag_matrix(double[::1] V)

cdef double _abs_min(double *a, Py_ssize_t n) noexcept nogil
cdef double _abs_diff_max(double *a, double *b, Py_ssize_t n) noexcept nogil
cdef double _mean(double *a, Py_ssize_t n) noexcept nogil
cdef double _std(double *a, double mu, Py_ssize_t n) noexcept nogil
cdef double _mad(double *a, double mu, Py_ssize_t n) noexcept nogil

cdef double quick_select(double *a, Py_ssize_t n) #noexcept nogil
# cdef double quick_select_t(double *a, Py_ssize_t n, Py_ssize_t step) #noexcept nogil
cdef double _median_1d(double[::1] x) #noexcept nogil
cdef void _median_2d(double[:,::1] x, double[::1] y) #noexcept nogil
cdef void _median_2d_t(double[:,::1] x, double[::1] y) #noexcept nogil
cdef void _median_absdev_2d(double[:,::1] x, double[::1] mu, double[::1] y) #noexcept nogil
cdef void _median_absdev_2d_t(double[:,::1] x, double[::1] mu, double[::1] y) #noexcept nogil
cdef void _robust_mean_2d(double[:,::1] x, double tau, double[::1] y) #noexcept nogil
cdef void _robust_mean_2d_t(double[:,::1] x, double tau, double[::1] y) #noexcept nogil

cdef double _kth_smallest(double *a, Py_ssize_t n, Py_ssize_t k) #noexcept nogil

cdef void _covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S) noexcept nogil
cdef void _covariance_matrix_weighted(
            double *X, const double *W, const double *loc, double *S, 
            const Py_ssize_t n, const Py_ssize_t N) noexcept nogil

cdef class RingArray:
    cdef public double[::1] data
    cdef Py_ssize_t size, index

    cpdef add(self, double val)
    cpdef mad(self)
