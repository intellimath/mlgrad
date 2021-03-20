# cython: language_level=3

from mlgrad.miscfuncs cimport init_rand, rand, fill

cdef class Batch:
    #
    cdef public Py_ssize_t n_samples, size
    cdef public Py_ssize_t[::1] indices
    #
    cdef void generate(self)
    cdef void init(self)

cdef class RandomBatch(Batch):
    pass
            
cdef class FixedBatch(Batch):
    pass

cdef class WholeBatch(Batch):
    pass
