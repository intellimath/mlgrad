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
    cpdef allocate(self, Py_ssize_t n):
        return None
    cpdef allocate2(self, Py_ssize_t n, Py_ssize_t m):
        return None
    cpdef get_allocated(self):
        return None
    cpdef Allocator suballocator(self):
        return self

cdef class ArrayAllocator(Allocator):

    def __init__(self, size, start=0, allocated=0):
        self.base = None
        self.size = size
        self.start = 0
        self.allocated = 0
        self.buf = np.zeros(size, 'd')
    #
    def __repr__(self):
        addr = 0
        if self.base is not None:
            addr = id(self.base)
        return "ArrayAllocator(%s %s %s %s)" % (addr, self.size, self.start, self.allocated)
    #
    cpdef allocate(self, Py_ssize_t n):
        cdef double[::1] ar
        cdef ArrayAllocator aa

        if n <= 0:
            raise RuntimeError('n <= 0')

        if self.allocated + n > self.size:
            raise RuntimeError('Memory out of buffer')

        ar = self.buf[self.allocated: self.allocated + n]
        self.allocated += n
       
        aa = self
        while aa.base is not None:
            aa.base.allocated = self.allocated
            aa = aa.base

        return ar
    #
    cpdef allocate2(self, Py_ssize_t n, Py_ssize_t m):
        cdef double[:,::1] ar2
        cdef ArrayAllocator aa
        cdef Py_ssize_t nm = n * m
       
        if n <= 0 or m <= 0:
            raise RuntimeError('n <= 0 or m <= 0')
       
        if self.allocated + nm > self.size:
            raise RuntimeError('Memory out of buffer')
        ar = self.buf[self.allocated: self.allocated + nm]
        ar2 = ar.reshape(n, m)
        self.allocated += nm
        
        aa = self
        while aa.base is not None:
            aa.base.allocated = self.allocated
            aa = aa.base
        return ar2
    #
    cpdef get_allocated(self):
        self.buf[self.start:self.allocated] = 0
        return self.buf[self.start: self.allocated]
    #
    cpdef Allocator suballocator(self):
        cdef ArrayAllocator allocator = ArrayAllocator.__new__(ArrayAllocator)

        allocator.buf = self.buf
        allocator.start = self.allocated
        allocator.allocated = self.allocated
        allocator.size = self.size
        allocator.base = self
        return allocator
