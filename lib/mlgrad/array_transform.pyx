

cdef void _array_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef double mu = inventory._mean(a, n)
    cdef double sigma = inventory._std(a, mu, n)

    for i in range(n):
        b[i] = (a[i] - mu) / sigma

def array_zscore(double[::1] a, double[::1] b):
    _array_zscore(&a[0], &b[0], a.shape[0])
    
cdef void _array_modified_zscore(double *a, double *b, Py_ssize_t n):
    cdef Py_ssize_t i
    cdef numpy.npy_intp nn = n
    cdef double[::1] aa = numpy.PyArray_EMPTY(1, &nn, numpy.NPY_DOUBLE, 0)
    cdef double mu = inventory._median_1d(&aa[0], n)
    cdef double sigma

    for i in range(n):
        aa[i] = fabs(aa[i] - mu)
    sigma = inventory._median_1d(&aa[0], n)

    for i in range(n):
        b[i] = 0.6745 * (a[i] - mu) / sigma

def array_modified_zscore(double[::1] a, double[::1] b):
    _array_modified_zscore(&a[0], &b[0], a.shape[0])

# cdef class ArrayTransformer:
#     cdef transform(self, double[::1] a):
#         pass

# cdef class ArrayNormalizer2(ArrayTransformer):
#     def __init__(self, i0=0):
#         self.i0 = i0

#     cdef transform(self, double[::1] a):
#         cdef Py_ssize_t i
#         cdef double s, v

#         s = 0
#         for i in range(self.i0, a.shape[0]):
#             v = a[i]
#             s += v*v

#         s = sqrt(s)

#         for i in range(a.shape[0]):
#             a[i] /= s
