# coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2020> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cimport cython

import numpy as np

cdef class Batch:
    #
    cdef void generate(self):
        pass
    
    cdef void init(self):
        pass
    
    def __len__(self):
        return self.size
    #
    def get_samples(self, X):
        return np.ascontiguousarray(X[self.indices])
    #
#     cdef void generate_samples2d(self, double[:,::1] X, double[:,::1] XX):
#         cdef Py_ssize_t k, j, i
#         cdef Py_ssize_t[::1] indices = self.indices
        
#         for j in range(self.size):
#             k = indices[j]
#             for i in range(X.shape[1]):
#                 XX[j,i] = X[k,i]
#     #
#     cdef void generate_samples1d(self, double[::1] Y, double[::1] YY):
#         cdef Py_ssize_t j, k
#         cdef Py_ssize_t[::1] indices = self.indices
        
#         for j in range(self.size):
#             k = indices[j]
#             YY[j] = Y[k]

cdef class RandomBatch(Batch):
    #
    def __init__(self, n_sample, size=None):
        self.n_sample = n_sample
        if size is None:
            self.size = self.n_sample
        else:
            self.size = size
        self.indices = np.zeros(self.size, dtype=np.intp)
    #
    cdef void init(self):
        init_rand()
    #
    cdef void generate(self):
        cdef Py_ssize_t i, k, size = self.size
        cdef Py_ssize_t n_sample = self.n_sample
        cdef Py_ssize_t[::1] indices = self.indices

        for i in range(size):
            k = rand(self.n_sample)
            indices[i] = k


# cdef class FixedBatch(Batch):
#     #
#     def __init__(self, indices):
#         self.size = len(indices)
#         self.n_sample = 0
#         self.indices = np.array(indices, dtype=np.intp)
#     #

cdef class WholeBatch(Batch):
    #
    def __init__(self, n_sample):
        self.size = int(n_sample)
        self.n_sample = self.size
        self.indices = np.arange(self.n_sample)
    #
    def get_samples(self, X):
        return X
    #
    # cdef void generate_samples2d(self, double[:,::1] X, double[:, ::1] XX):
    #     if &XX[0,0] != &X[0,0]:
    #         Batch.generate_samples2d(self, X, XX)
    # #
    # cdef void generate_samples1d(self, double[::1] Y, double[::1] YY):
    #     if &YY[0] != &Y[0]:
    #         Batch.generate_samples1d(self, Y, YY)

def make_batch(n_sample, size=None):
    if size is None:
        return WholeBatch(n_sample)
    else:
        return RandomBatch(n_sample, size)
