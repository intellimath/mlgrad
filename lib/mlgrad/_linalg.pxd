cimport cython

from libc.string cimport memcpy, memset


cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil
cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil
cdef void clear(double[::1] to) noexcept nogil
cdef void clear2(double[:,::1] to) noexcept nogil

cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil
cdef void fill(double[::1] to, const double c) noexcept nogil

cdef void _move(double*, const double*, const Py_ssize_t) noexcept nogil
cdef void _move_t(double *to, const double *src, const Py_ssize_t n, const Py_ssize_t step) noexcept nogil
cdef void move(double[::1] to, double[::1] src) noexcept nogil
cdef void move2(double[:,::1] to, double[:,::1] src) noexcept nogil
cdef void move3(double[:,:,::1] to, double[:,:,::1] src) noexcept nogil

cdef void _add(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void add(double[::1] c, double[::1] a, double[::1] b) noexcept nogil

cdef void _iadd(double *a, const double *b, const Py_ssize_t n) noexcept nogil
cdef void _add(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil

cdef void sub(double[::1] c, double[::1] a, double[::1] b) noexcept nogil
cdef void _sub(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil

cdef void _isub(double *a, const double *b, const Py_ssize_t n) noexcept nogil
# cdef void _isub_mask(double *a, const double *b, uint8 *m, const Py_ssize_t n) noexcept nogil
