
cdef void _array_zscore(double *a, double *b, Py_ssize_t n)
cdef void _array_modified_zscore(double *a, double *b, Py_ssize_t n)
cdef void _array_modified_zscore_mu(double *a, double *b, Py_ssize_t n, double mu)
cdef void _array_diff4(double *x, double *y, const Py_ssize_t n)
cdef void _array_diff3(double *x, double *y, const Py_ssize_t n)
cdef void _array_diff2(double *x, double *y, const Py_ssize_t n)
cdef void _array_diff1(double *x, double *y, const Py_ssize_t n)


# cdef class ArrayTransformer:
#     cdef transform(self, double[::1] a)


# cdef class ArrayNormalizer2(ArrayTransformer):
#     cdef Py_ssize_t i0
#     cdef transform(self, double[::1] a)
