# coding: utf-8

# The MIT License (MIT)

# Copyright (c) «2015-2021» «Shibzukhov Zaur, szport at gmail dot com»

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software - recordclass library - and associated documentation files 
# (the "Software"), to deal in the Software without restriction, including 
# without limitation the rights to use, copy, modify, merge, publish, distribute, 
# sublicense, and/or sell copies of the Software, and to permit persons to whom 
# the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

cimport cython
import numpy as np

cdef inline Py_ssize_t resize(Py_ssize_t size):
    if size < 9:
        return size + (size // 8) + 3
    else:
        return size + (size // 8) + 6

# cdef list_values empty_list(Py_ssize_t size, Py_ssize_t itemsize):
#     cdef list_values op

#     op = <list_values>list_values.__new__(list_values, None)
#     op.data = <void*>PyMem_Malloc(size*itemsize)
#     op.size = op.allocated = size        
#     return op

# def new_list_values(*args, ):
#     cdef Py_ssize_t i, size = Py_SIZE(args)
#     cdef list_values op = <list_values>list_values.__new__(list_values, None)
#     cdef double *data;
#     cdef PyObject *v;
    
#     op = empty_list(size, )
#     op.size = op.allocated = size
#     data = op.data = <double*>PyMem_Malloc(size*sizeof(double))
#     for i in range(size):
#         v = PyTuple_GET_ITEM(<PyObject*>args, i)
#         if Py_TYPE(v) is &PyFloat_Type:
#             data[i] = PyFloat_AS_double(<object>v);
#         else:
#             raise TypeError("This object is not a double")
        
#     return <list_values>op

# sizeof_double = sizeof(double)
# sizeof_pdouble = sizeof(double*)
# sizeof_int = sizeof(int)
# sizeof_pint = sizeof(int*)

            
@cython.no_gc
cdef class list_double:
    
    def __cinit__(self, size=0):
        cdef Py_ssize_t _size = size

        self.data = data = <double*>PyMem_Malloc(_size*sizeof(double))
        self.size = self.allocated = _size

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.size
    
    cdef inline double _get(self, Py_ssize_t i):
        return self.data[i]

    cdef inline void _set(self, Py_ssize_t i, double v):
        self.data[i] = v
        
    def __getitem__(self, i):
        cdef Py_ssize_t ii = i

        if ii < 0:
            ii = self.size + ii
        if 0 <= ii < self.size:
            return self._get(ii)
        else:
            raise IndexError('invalid index %s' % i)

    def __setitem__(self, i, v):
        cdef Py_ssize_t ii = i
        cdef double vv = v

        if ii < 0:
            ii = self.size + ii
        if 0 <= ii < self.size:
            self._set(i, vv)
        else:
            raise IndexError('invalid index %s' % i)

    cdef void _append(self, double op):
        cdef Py_ssize_t size, newsize
        
        size = self.size
        if size >= self.allocated:
            newsize = resize(size + 1)
            self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize        
        self.data[size] = op;
        self.size += 1
        
    def append(self, v):
        cdef double vv = v
        self._append(vv)
        
    cdef void _extend(self, double *op, Py_ssize_t n):
        cdef Py_ssize_t i, newsize, size
        
        size = self.size
        if size + n >= self.allocated:
            newsize = resize(size + n)
            self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize
        for i in range(n):
            self.data[size + i] = op[i]
        self.size += n
        
    def extend(self, ops):
        cdef double vv
        for v in ops:
            vv = v
            self._append(v)
            
    def copy(self):
        cdef Py_ssize_t i, size = self.size
        cdef list_double cp = list_double(size)
        
        for i in range(size):
            cp._set(i, self._get(i))
        
        return cp
    
    def as_list(self):
        cdef Py_ssize_t i, size = self.size
        cdef list res = []
        
        for i in range(size):
            res.append(self.get_double(i))
        return res
    
    def as_nparray(self):
        cdef Py_ssize_t i, size = self.size
        cdef double[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get(i)
        return res
    
    def as_memview(self):
        cdef Py_ssize_t i, size = self.size
        cdef double[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get(i)
        return data

@cython.no_gc
cdef class list_int:
    
    def __cinit__(self, size=0):
        cdef Py_ssize_t _size = size

        self.data = data = <int*>PyMem_Malloc(_size*sizeof(int))
        self.size = self.allocated = _size

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.size
    
    cdef inline int _get(self, Py_ssize_t i):
        return self.data[i]

    cdef inline void _set(self, Py_ssize_t i, int v):
        self.data[i] = v
        
    def __getitem__(self, i):
        cdef Py_ssize_t ii = i

        if ii < 0:
            ii = self.size + ii
        if 0 <= ii < self.size:
            return self._get(ii)
        else:
            raise IndexError('invalid index %s' % i)

    def __setitem__(self, i, v):
        cdef Py_ssize_t ii = i
        cdef int vv = v

        if ii < 0:
            ii = self.size + ii
        if 0 <= ii < self.size:
            self._set(i, vv)
        else:
            raise IndexError('invalid index %s' % i)

    cdef void _append(self, int op):
        cdef Py_ssize_t size, newsize
        
        size = self.size
        if size >= self.allocated:
            newsize = resize(size + 1)
            self.data = <int*>PyMem_Realloc(self.data, newsize*sizeof(int))
            self.allocated = newsize        
        self.data[size] = op;
        self.size += 1
        
    def append(self, v):
        cdef int vv = v
        self._append(vv)
        
    cdef void _extend(self, int *op, Py_ssize_t n):
        cdef Py_ssize_t i, newsize, size
        
        size = self.size
        if size + n >= self.allocated:
            newsize = resize(size + n)
            self.data = <int*>PyMem_Realloc(self.data, newsize*sizeof(int))
            self.allocated = newsize
        for i in range(n):
            self.data[size + i] = op[i]
        self.size += n
        
    def extend(self, ops):
        cdef int vv
        for v in ops:
            vv = v
            self._append(v)
            
    def copy(self):
        cdef Py_ssize_t i, size = self.size
        cdef list_int cp = list_int(size)
        
        for i in range(size):
            cp._set(i, self._get(i))
        
        return cp
    
    def as_list(self):
        cdef Py_ssize_t i, size = self.size
        cdef list res = []
        
        for i in range(size):
            res.append(self._get(i))
        return res
    
    def as_nparray(self):
        cdef Py_ssize_t i, size = self.size
        cdef int[::1] data
        
        res = np.empty(size, 'i')
        data = res
        for i in range(size):
            data[i] = self._get(i)
        return res
    
    def as_memview(self):
        cdef Py_ssize_t i, size = self.size
        cdef int[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get(i)
        return data
    