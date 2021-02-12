# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False


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

import numpy as np

cdef class Batch:
    #
    cdef void generate(self):
        pass
    
    def __len__(self):
        return self.size

cdef class RandomBatch(Batch):
    #
    def __init__(self, n_samples, size=None):
        self.n_samples = n_samples
        if size is None:
            self.size = self.n_samples
        else:
            self.size = size
        self.indices = np.zeros(self.size, dtype='i')
        init_rand()
    #
    cdef void generate(self):
        cdef Py_ssize_t i, k, size = self.size
        cdef Py_ssize_t n_samples = self.n_samples
        cdef Py_ssize_t[::1] indices = self.indices
        
        for i in range(size):
            k = rand(self.n_samples)
            indices[i] = k
            
cdef class FixedBatch(Batch):
    #
    def __init__(self, indices):
        self.size = len(indices)
        self.n_samples = 0
        self.indices = indices
    #

cdef class WholeBatch(Batch):
    #
    def __init__(self, n_samples):
        self.size = int(n_samples)
        self.n_samples = self.size
        self.indices = np.arange(n_samples, dtype='l')
    #

def make_batch(n_samples, size=None):
    if size is None:
        return WholeBatch(n_samples)
    else:
        return RandomBatch(n_samples, size)

def make_batch_from(indices):
    return FixedBatch(indices)