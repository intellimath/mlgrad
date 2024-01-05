# coding: utf-8 

# The MIT License (MIT)
#
# Copyright (c) <2015-2021> <Shibzukhov Zaur, szport at gmail dot com>
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

cdef class Allocator(object):
    #
    def allocate(self, Py_ssize_t n):
        return None
    def allocate2(self, Py_ssize_t n, Py_ssize_t m):
        return None
    def get_allocated(self):
        return None
    def suballocator(self):
        return self

cdef class ArrayAllocator(Allocator):

    def __init__(self, size):
        self.base = None
        self.size = size
        self.start = 0
        self.n_allocated = 0
        self.buf = np.zeros(size, 'd')
    #
    def __repr__(self):
        addr = 0
        if self.base is not None:
            addr = id(self.base)
        return "ArrayAllocator(%s %s %s %s)" % (addr, self.size, self.start, self.n_allocated)
    #
    def allocate(self, Py_ssize_t n):
        cdef ArrayAllocator aa

        if n <= 0:
            raise RuntimeError('n <= 0')

        if self.n_allocated + n > self.size:
            raise RuntimeError('Memory out of buffer')

        ar = self.buf[self.n_allocated: self.n_allocated + n]
        self.n_allocated += n

        aa = self
        while aa.base is not None:
            aa.base.n_allocated = self.n_allocated
            aa = aa.base

        return ar
    #
    def allocate2(self, Py_ssize_t n, Py_ssize_t m):
        cdef ArrayAllocator aa
        cdef Py_ssize_t nm = n * m

        if n <= 0 or m <= 0:
            raise RuntimeError('n <= 0 or m <= 0')

        if self.n_allocated + nm > self.size:
            raise RuntimeError('Memory out of buffer')
        ar = self.buf[self.n_allocated: self.n_allocated + nm]
        ar2 = ar.reshape((n, m))
        self.n_allocated += nm
        
        aa = self
        while aa.base is not None:
            aa.base.n_allocated = self.n_allocated
            aa = aa.base

        return ar2
    #
    def get_allocated(self):
        self.buf[self.start:self.n_allocated] = 0
        return self.buf[self.start: self.n_allocated]
    #
    def suballocator(self):
        cdef ArrayAllocator allocator = ArrayAllocator.__new__(ArrayAllocator)

        allocator.buf = self.buf
        allocator.start = self.n_allocated
        allocator.n_allocated = self.n_allocated
        allocator.size = self.size
        allocator.base = self
        return allocator
