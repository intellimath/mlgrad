# coding: utf-8 

# cython: language_level=3

cdef int get_num_threads() nogil
cdef void set_num_threads(int num) nogil

cdef void _fill(double *to, const double c, const Py_ssize_t n) nogil
cdef double _conv(const double*, const double*, const Py_ssize_t) nogil
cdef void _move(double*, const double*, const Py_ssize_t) nogil
cdef double _sum(const double*, const Py_ssize_t) nogil
cdef void _mul_add_array(double *a, const double *b, double c, const Py_ssize_t n) nogil
cdef void _mul_const(double *a, const double c, const Py_ssize_t n) nogil
cdef void _matdot(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) nogil
cdef void _matdot2(double*, double*, const double*, const Py_ssize_t, const Py_ssize_t) nogil
cdef void _mul_add_arrays(double *a, double *M, const double *ss, const Py_ssize_t n_input, const Py_ssize_t n_output) nogil
cdef void _mul_grad(double *grad, const double *X, const double *ss, const Py_ssize_t n_input, const Py_ssize_t n_output) nogil

cdef void fill(double[::1] to, const double c) nogil
cdef void move(double[::1] to, double[::1] src) nogil
cdef double conv(double[::1] a, double[::1] b) nogil
cdef double sum(double[::1] a) nogil
cdef void mul_const(double[::1] a, const double c) nogil
cdef void mul_add_array(double[::1] a, double[::1] b, double c) nogil
cdef void matdot(double[::1] output, double[:,::1] M, double[::1] X) nogil
cdef void matdot2(double[::1] output, double[:,::1] M, double[::1] X) nogil
cdef void mul_add_arrays(double[::1] a, double[:,::1] M, double[::1] ss) nogil
cdef void mul_grad(double[:,::1] grad, double[::1] X, double[::1] ss) nogil

