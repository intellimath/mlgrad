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

from cpython.buffer cimport PyObject_GetBuffer, PyBuffer_Release, \
                            PyBUF_ANY_CONTIGUOUS, PyBUF_SIMPLE
 
@cython.binding(True)
cdef (double*, int, int) _memoryview_start_and_len(double[::1] m):
    '''Gets a slice object that corresponds to the given memoryview's view on the undelying.'''
    cdef Py_buffer view_buffer, underlying_buffer
    PyObject_GetBuffer(m, &view_buffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
    try:
        view_ptr = <const char *>view_buffer.buf
        PyObject_GetBuffer(m.from_object, &underlying_buffer, PyBUF_SIMPLE | PyBUF_ANY_CONTIGUOUS)
        try:
            underlying_ptr = <const char *>underlying_buffer.buf
            if view_ptr < underlying_ptr:
                raise RuntimeError("Weird: view_ptr < underlying_ptr")
            start = view_ptr - underlying_ptr
            return (<double*>underlying_ptr, start, len(m))
        finally:
            PyBuffer_Release(&underlying_buffer)
    finally:
        PyBuffer_Release(&view_buffer)

def memoryview_start_and_len(m):
    cdef (double*, int, int) stat = _memoryview_start_and_len(m)
    return (<int>stat[0], stat[1], stat[2])
    
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
        self.start = self.i = 0
        self.n_allocated = 0
        self.buf = np.zeros(size, 'd')
        self.buf_array = self.buf
    #
    def __repr__(self):
        addr = 0
        if self.base is not None:
            addr = id(self.base)
        return "ArrayAllocator(%s %s %s %s)" % (addr, self.size, self.start, self.n_allocated)
    #
    def close(self):
        if self.base is not None:
            self.base.n_allocated = self.n_allocated
    #
    def allocate(self, Py_ssize_t n):
        cdef ArrayAllocator aa

        if n <= 0:
            raise RuntimeError('n <= 0')

        if self.n_allocated + n > self.size:
            raise RuntimeError('Memory out of buffer')

        self.i = self.n_allocated
        self.n_allocated += n
        ar = self.buf[self.i: self.n_allocated]

        # aa = self
        # while aa.base is not None:
        #     aa.base.n_allocated = self.n_allocated
        #     aa = aa.base

        return ar
    #
    def allocate2(self, Py_ssize_t n, Py_ssize_t m):
        # cdef ArrayAllocator aa
        cdef Py_ssize_t nm = n * m

        if n <= 0 or m <= 0:
            raise RuntimeError('n <= 0 or m <= 0')

        if self.n_allocated + nm > self.size:
            raise RuntimeError('Memory out of buffer')

        self.i = self.n_allocated
        self.n_allocated += nm
        ar = self.buf[self.i: self.n_allocated]

        ar2 = ar.reshape((n, m))
        
        # aa = self
        # while aa.base is not None:
        #     aa.base.n_allocated = self.n_allocated
        #     aa = aa.base

        return ar2
    #
    def get_allocated(self):
        self.buf[self.start:self.n_allocated] = 0
        return self.buf[self.start: self.n_allocated]
    #
    def suballocator(self):
        cdef ArrayAllocator allocator = ArrayAllocator.__new__(ArrayAllocator)

        allocator.buf = self.buf
        allocator.buf_array = self.buf_array
        allocator.start = allocator.i = self.n_allocated
        allocator.n_allocated = self.n_allocated
        allocator.size = self.size
        allocator.base = self
        return allocator
