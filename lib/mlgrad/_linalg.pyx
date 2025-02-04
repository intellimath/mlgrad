
# clear

cdef void _clear(double *to, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = 0
        
cdef void clear(double[::1] to) noexcept nogil:
    _clear(&to[0], to.shape[0])

cdef void _clear2(double *to, const Py_ssize_t n, const Py_ssize_t m) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n*m):
        to[i] = 0
        
cdef void clear2(double[:,::1] to) noexcept nogil:
    _clear2(&to[0,0], to.shape[0], <const Py_ssize_t>to.shape[1])

# fill
    
cdef void _fill(double *to, const double c, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = c

cdef void fill(double[::1] to, const double c) noexcept nogil:
    _fill(&to[0], c, to.shape[0])

# move

cdef void _move_t(double *to, const double *src, const Py_ssize_t n, const Py_ssize_t step) noexcept nogil:
    cdef Py_ssize_t i, j
    j = 0
    for i in range(n):
        to[i] = src[j]
        j += step
    
cdef void _move(double *to, const double *src, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        to[i] = src[i]

cdef void move(double[::1] to, double[::1] src) noexcept nogil:
    _move(&to[0], &src[0], to.shape[0])
    
cdef void move2(double[:, ::1] to, double[:,::1] src) noexcept nogil:
    _move(&to[0,0], &src[0,0], to.shape[0] * to.shape[1])

cdef void move3(double[:,:,::1] to, double[:,:,::1] src) noexcept nogil:
    _move(&to[0,0,0], &src[0,0,0], to.shape[0] * to.shape[1] * to.shape[2])

# add

cdef void _add(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        c[i] = a[i] + b[i]

cdef void add(double[::1] c, double[::1] a, double[::1] b) noexcept nogil:
    _add(&c[0], &a[0], &b[0], a.shape[0])

# iadd

cdef void _iadd(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        a[i] += b[i]

cdef void iadd(double[::1] a, double[::1] b) noexcept nogil:
    _iadd(&a[0], &b[0], a.shape[0])

# sub

cdef void _sub(double *c, const double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        c[i] = a[i] - b[i]
    
cdef void sub(double[::1] c, double[::1] a, double[::1] b) noexcept nogil:
    _sub(&c[0], &a[0], &b[0], a.shape[0])


# isub

cdef void _isub(double *a, const double *b, const Py_ssize_t n) noexcept nogil:
    cdef Py_ssize_t i
    for i in range(n):
        a[i] -= b[i]

cdef void isub(double[::1] a, double[::1] b) noexcept nogil:
    _isub(&a[0], &b[0], a.shape[0])
