# cython: language_level=3

from mlgrad.miscfuncs cimport init_rand, rand, fill

cdef class Batch:
    #
    cdef public int n_samples, size
    cdef public int[::1] indices
    #
    cdef void generate(self)

cdef class RandomBatch(Batch):
    pass
            
cdef class FixedBatch(Batch):
    pass

cdef class WholeBatch(Batch):
    pass
