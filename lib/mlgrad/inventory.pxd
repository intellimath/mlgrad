# coding: utf-8 

# cython: language_level=3

cimport cython

from numpy cimport npy_uint8 as uint8
cimport numpy

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
    object PyFloat_FromDouble(double)

cdef public double _double_max
cdef public double _double_min

cdef void init_rand() noexcept nogil
cdef long rand(long N) noexcept nogil

cdef int get_num_threads() noexcept nogil
cdef int get_num_procs() noexcept nogil
cdef int get_num_threads_ex(int n) noexcept nogil
cdef int get_num_procs_ex(int n) noexcept nogil
# cdef void set_num_threads(int num) noexcept nogil

cdef bint _iscontiguousarray(object ob)
cdef bint _isnumpyarray(object ob)

cdef object _asarray(object ob)

cdef int _hasnan(double *a, const Py_ssize_t n) noexcept nogil

cdef double _min(double *a, Py_ssize_t n) noexcept nogil

cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil
cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil
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
# cdef double conv(double[::1] a, double[::1] b) noexcept nogil
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
cdef double dot_t(double[::1] a, double[:,::1] b) noexcept nogil
cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil
cdef void matdot2(double[::1] output, double[:,::1] M, double[::1] X) noexcept nogil
cdef void mul_add_arrays(double[::1] a, double[:,::1] M, double[::1] ss) noexcept nogil
cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) noexcept nogil
cdef void normalize(double[::1] a) noexcept nogil
cdef void normalize2(double[::1] a) noexcept nogil

cdef _mahalanobis_norm(double[:,::1] S, double[:,::1] X, double[::1] Y)
cdef _mahalanobis_distance(double[:,::1] S, double[:,::1] X, double[::1] c, double[::1] Y)

cdef void _scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil
cdef void _scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil
cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) noexcept nogil

cdef object empty_array(Py_ssize_t size)
cdef object empty_array2(Py_ssize_t size1, Py_ssize_t size2)
cdef object zeros_array(Py_ssize_t size)
cdef object zeros_array2(Py_ssize_t size1, Py_ssize_t size2)

cdef object filled_array(Py_ssize_t size, double val)

cdef object diag_matrix(double[::1] V)

cdef double _abs_min(double *a, Py_ssize_t n) noexcept nogil
cdef double _abs_diff_max(double *a, double *b, Py_ssize_t n) noexcept nogil
cdef double _mean(double *a, Py_ssize_t n) noexcept nogil
cdef double _std(double *a, double mu, Py_ssize_t n) noexcept nogil
cdef double _mad(double *a, double mu, Py_ssize_t n) noexcept nogil

cdef void _zscore(double *a, double *b, Py_ssize_t n)
cdef void _modified_zscore(double *a, double *b, Py_ssize_t n)
cdef void _modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu)
cdef void _diff4(double *x, double *y, const Py_ssize_t n) noexcept nogil
cdef void _diff3(double *x, double *y, const Py_ssize_t n) noexcept nogil
cdef void _diff2(double *x, double *y, const Py_ssize_t n) noexcept nogil
cdef void _diff2w2(double *x, double *w, double *y, const Py_ssize_t n) noexcept nogil
cdef void _diff1(double *x, double *y, const Py_ssize_t n) noexcept nogil
cdef void _relative_max(double *x, double *y, const Py_ssize_t n) noexcept nogil
cdef void _relative_abs_max(double *x, double *y, const Py_ssize_t n) noexcept nogil


# /*
#  *  This Quickselect routine is based on the algorithm described in
#  *  "Numerical recipes in C", Second Edition,
#  *  Cambridge University Press, 1992, Section 8.5, ISBN 0-521-43108-5
#  *  This code by Nicolas Devillard - 1998. Public domain.
#  */
    
# #define ELEM_SWAP(a,b) { register double t=(a);(a)=(b);(b)=t; }

# static double quick_select(double *arr, Py_ssize_t n) 
# {
#     Py_ssize_t low, high ;
#     Py_ssize_t median;
#     Py_ssize_t middle, ll, hh;

#     low = 0 ; high = n-1 ; median = (low + high) / 2;
#     for (;;) {
#         if (high <= low) /* One element only */
#             return arr[median] ;

#         if (high == low + 1) {  /* Two elements only */
#             if (arr[low] > arr[high])
#                 ELEM_SWAP(arr[low], arr[high]) ;
#             return arr[median] ;
#         }

#         /* Find median of low, middle and high items; swap into position low */
#         middle = (low + high) / 2;
#         if (arr[middle] > arr[high])    ELEM_SWAP(arr[middle], arr[high]) ;
#         if (arr[low] > arr[high])       ELEM_SWAP(arr[low], arr[high]) ;
#         if (arr[middle] > arr[low])     ELEM_SWAP(arr[middle], arr[low]) ;
    
#         /* Swap low item (now in position middle) into position (low+1) */
#         ELEM_SWAP(arr[middle], arr[low+1]) ;
    
#         /* Nibble from each end towards middle, swapping items when stuck */
#         ll = low + 1;
#         hh = high;
#         for (;;) {
#             do ll++; while (arr[low] > arr[ll]) ;
#             do hh--; while (arr[hh]  > arr[low]) ;
    
#             if (hh < ll)
#                 break;
    
#             ELEM_SWAP(arr[ll], arr[hh]) ;
#         }
    
#         /* Swap middle item (in position low) back into correct position */
#         ELEM_SWAP(arr[low], arr[hh]) ;
    
#         /* Re-set active partition */
#         if (hh <= median)
#             low = ll;
#         if (hh >= median)
#             high = hh - 1;
#     }
# }

# #undef ELEM_SWAP

cdef double quick_select(double *a, Py_ssize_t n) #noexcept nogil

# cdef double quick_select_t(double *a, Py_ssize_t n, Py_ssize_t step) #noexcept nogil
cdef double _median_1d(double[::1] x) #noexcept nogil
cdef double _median_absdev_1d(double[::1] x, double mu) #noexcept nogil
cdef void _median_2d(double[:,::1] x, double[::1] y) #noexcept nogil
cdef void _median_2d_t(double[:,::1] x, double[::1] y) #noexcept nogil
cdef void _median_absdev_2d(double[:,::1] x, double[::1] mu, double[::1] y) #noexcept nogil
cdef void _median_absdev_2d_t(double[:,::1] x, double[::1] mu, double[::1] y) #noexcept nogil
cdef double _robust_mean_1d(double[::1] x, double tau) #noexcept nogil
cdef void _robust_mean_2d(double[:,::1] x, double tau, double[::1] y) #noexcept nogil
cdef void _robust_mean_2d_t(double[:,::1] x, double tau, double[::1] y) #noexcept nogil

cdef double _kth_smallest(double *a, Py_ssize_t n, Py_ssize_t k) noexcept nogil

cdef void _covariance_matrix(double[:, ::1] X, double[::1] loc, double[:,::1] S) noexcept nogil
cdef void _covariance_matrix_weighted(double[:, ::1] X, double[::1] W, 
                                      double[::1] loc, double[:,::1] S) noexcept nogil
# cdef void _covariance_matrix_weighted(
#             double *X, const double *W, const double *loc, double *S, 
#             const Py_ssize_t n, const Py_ssize_t N) noexcept nogil

cdef _inverse_matrix(double[:,::1] AM, double[:,::1] IM)


cdef class RingArray:
    cdef public double[::1] data
    cdef Py_ssize_t size, index

    cpdef add(self, double val)
    cpdef mad(self)

cdef double _norm2(double[::1] a)
