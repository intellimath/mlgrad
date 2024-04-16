# coding: utf-8 

# cython: language_level=3

cimport cython

from numpy cimport npy_uint8 as uint8

from libc.string cimport memcpy, memset
from libc.math cimport isnan

cdef int get_num_threads() noexcept nogil
cdef int get_num_procs() noexcept nogil
cdef int get_num_threads_ex(int n) noexcept nogil
cdef int get_num_procs_ex(int n) noexcept nogil
# cdef void set_num_threads(int num) noexcept nogil

cdef int _hasnan(double *a, const Py_ssize_t n) noexcept nogil

cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil
cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil
cdef double _conv(const double*, const double*, const Py_ssize_t) noexcept nogil
cdef void _move(double*, const double*, const Py_ssize_t) noexcept nogil
cdef double _sum(const double*, const Py_ssize_t) noexcept nogil
cdef void _add(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _sub(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _sub_mask(double *a, const double *b, uint8 *m, const Py_ssize_t n) noexcept nogil
cdef void _mul(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _mul_add(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_set(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_set1(double *a, const double *b, const double c, const Py_ssize_t n) noexcept nogil
cdef void _mul_const(double *a, const double c, const Py_ssize_t n) noexcept nogil
cdef double _dot1(const double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef double _dot(const double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef double _dot_t(const double *a, double *b, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void _matdot(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) noexcept nogil
cdef void _matdot2(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) noexcept nogil
cdef void _mul_add_arrays(double *a, double *M, const double *ss, const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil
cdef void _mul_grad(double *grad, const double *X, const double *ss, const Py_ssize_t n_input, const Py_ssize_t n_output) noexcept nogil
cdef void _multiply(double *a, const double *b, const double *x, const Py_ssize_t n) noexcept nogil
cdef void _normalize(double *a, const Py_ssize_t n) noexcept nogil


cdef int hasnan(double[::1] a) noexcept nogil

cdef void clear(double[::1] to) noexcept nogil
cdef void clear2(double[:,::1] to) noexcept nogil
cdef void fill(double[::1] to, const double c) noexcept nogil
cdef void move(double[::1] to, double[::1] src) noexcept nogil
cdef void move2(double[:,::1] to, double[:,::1] src) noexcept nogil
cdef void move3(double[:,:,::1] to, double[:,:,::1] src) noexcept nogil
cdef double conv(double[::1] a, double[::1] b) noexcept nogil
cdef double sum(double[::1] a) noexcept nogil
cdef void add(double[::1] a, double[::1] b) noexcept nogil
cdef void add2(double[:,::1] a, double[:,::1] b) noexcept nogil
cdef void sub(double[::1] a, double[::1] b) noexcept nogil
cdef void sub_mask(double[::1] a, double[::1] b, uint8[::1] m) noexcept nogil
cdef void mul_const(double[::1] a, const double c) noexcept nogil
cdef void mul_const2(double[:, ::1] a, const double c) noexcept nogil
cdef void mul_const3(double[:,:,::1] a, const double c) noexcept nogil
cdef void mul(double[::1] a, double[::1] b) noexcept nogil
cdef void mul2(double[:,::1] a, double[:,::1] b) noexcept nogil
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
cdef void multiply(double[::1] a, double[::1] b, double[::1] c) noexcept nogil
cdef void normalize(double[::1] a) noexcept nogil

cdef void scatter_matrix_weighted(double[:,::1] X, double[::1] W, double[:,::1] S) noexcept nogil
cdef void scatter_matrix(double[:,::1] X, double[:,::1] S) noexcept nogil
cdef void weighted_sum_rows(double[:,::1] X, double[::1] W, double[::1] Y) noexcept nogil
