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

# @cython.no_gc
# cdef class list_values:
    
#     def __cinit__(self, Py_ssize_t itemsize, Py_ssize_t size=0):

#         self.data = data = <double*>PyMem_Malloc(size*sizeof(itemsize))
#         self.size = self.allocated = size

#     def __dealloc__(self):
#         PyMem_Free(self.data)

#     def __len__(self):
#         return self.size
        
#     cdef inline double* as_double_array(self):
#         return <double*>self.data

#     cdef inline double** as_pdouble_array(self):
#         return <double**>self.data
    
#     cdef inline double _get_double(self, Py_ssize_t i):
#         return (<double*>self.data)[i]

#     cdef inline double* _get_pdouble(self, Py_ssize_t i):
#         return (<double**>self.data)[i]
    
#     cdef inline void _set_double(self, Py_ssize_t i, double p):
#         (<double*>self.data)[i] = p

#     cdef inline void _set_pdouble(self, Py_ssize_t i, double *p):
#         (<double**>self.data)[i] = p
        
#     def get_double(self, i):
#         if i < self.size:
#             return self._get_double(i)
#         else:
#             raise IndexError("invalid index " + str(i))

#     def set_double(self, i, v):
#         return self._set_double(i, v)

#     cdef void _append_double(self, double op):
#         cdef Py_ssize_t size, newsize
        
#         size = self.size
#         if size >= self.allocated:
#             newsize = resize(size + 1)
#             self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double))
#             self.allocated = newsize        
#         (<double*>self.data)[size] = op;
#         self.size += 1

#     cdef void _append_pdouble(self, double *op):
#         cdef Py_ssize_t size, newsize
        
#         size = self.size
#         if size >= self.allocated:
#             newsize = resize(size + 1)
#             self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double*))
#             self.allocated = newsize        
#         (<double**>self.data)[size] = op;
#         self.size += 1
        
#     def append_double(self, v):
#         cdef double dv = v
#         self._append_double(dv)
        
#     cdef void _extend_double(self, double *op, Py_ssize_t n):
#         cdef Py_ssize_t i, newsize, size
        
#         size = self.size
#         if size + n >= self.allocated:
#             newsize = resize(size + n)
#             self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double))
#             self.allocated = newsize
#         for i in range(n):
#             (<double*>self.data)[size + i] = op[i]
#         self.size += n

#     cdef void _extend_pdouble(self, double **op, Py_ssize_t n):
#         cdef Py_ssize_t i, newsize, size
        
#         size = self.size
#         if size + n >= self.allocated:
#             newsize = resize(size + n)
#             self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double*))
#             self.allocated = newsize
#         for i in range(n):
#             (<double**>self.data)[size + i] = op[i]
#         self.size += n
        
#     def extend_double(self, ops):
#         for v in ops:
#             self._append_double(v)
            
#     def as_list_double(self):
#         cdef Py_ssize_t i, size = self.size
#         cdef list res = []
        
#         for i in range(size):
#             res.append(self.get_double(i))
#         return res
    
#     def as_nparray_double(self):
#         cdef Py_ssize_t i, size = self.size
#         cdef double[::1] data
        
#         res = np.empty(size, 'd')
#         data = res
#         for i in range(size):
#             data[i] = self._get_double(i)
#         return res
    
#     def as_memview_double(self):
#         cdef Py_ssize_t i, size = self.size
#         cdef double[::1] data
        
#         res = np.empty(size, 'd')
#         data = res
#         for i in range(size):
#             data[i] = self._get_double(i)
#         return data
            
@cython.no_gc
cdef class list_doubles:
    
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
        cdef list_doubles cp = list_doubles(size)
        
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
    
    def as_memview_double(self):
        cdef Py_ssize_t i, size = self.size
        cdef double[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get(i)
        return data
