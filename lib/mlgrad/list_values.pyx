# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False
 
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

sizeof_double = sizeof(double)
sizeof_pdouble = sizeof(double*)
sizeof_int = sizeof(int)
sizeof_pint = sizeof(int*)

@cython.no_gc
cdef class list_values:
    
    def __cinit__(self, Py_ssize_t itemsize, Py_ssize_t size=0):

        self.data = data = <double*>PyMem_Malloc(size*sizeof(itemsize))
        self.size = self.allocated = size

    def __dealloc__(self):
        PyMem_Free(self.data)

    def __len__(self):
        return self.size
        
    cdef inline double* as_double_array(self):
        return <double*>self.data

    cdef inline double** as_pdouble_array(self):
        return <double**>self.data
    
    cdef inline double _get_double(self, Py_ssize_t i):
        return (<double*>self.data)[i]

    cdef inline double* _get_pdouble(self, Py_ssize_t i):
        return (<double**>self.data)[i]
    
    cdef inline void _set_double(self, Py_ssize_t i, double p):
        (<double*>self.data)[i] = p

    cdef inline void _set_pdouble(self, Py_ssize_t i, double *p):
        (<double**>self.data)[i] = p
        
    def get_double(self, i):
        if i < self.size:
            return self._get_double(i)
        else:
            raise IndexError("invalid index " + str(i))

    def set_double(self, i, v):
        return self._set_double(i, v)

    cdef void _append_double(self, double op):
        cdef Py_ssize_t size, newsize
        
        size = self.size
        if size >= self.allocated:
            newsize = resize(size + 1)
            self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize        
        (<double*>self.data)[size] = op;
        self.size += 1

    cdef void _append_pdouble(self, double *op):
        cdef Py_ssize_t size, newsize
        
        size = self.size
        if size >= self.allocated:
            newsize = resize(size + 1)
            self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double*))
            self.allocated = newsize        
        (<double**>self.data)[size] = op;
        self.size += 1
        
    def append_double(self, v):
        self._append_double(v)
        
    cdef void _extend_double(self, double *op, Py_ssize_t n):
        cdef Py_ssize_t i, newsize, size
        
        size = self.size
        if size + n >= self.allocated:
            newsize = resize(size + n)
            self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize
        for i in range(n):
            (<double*>self.data)[size + i] = op[i]
        self.size += n

    cdef void _extend_pdouble(self, double **op, Py_ssize_t n):
        cdef Py_ssize_t i, newsize, size
        
        size = self.size
        if size + n >= self.allocated:
            newsize = resize(size + n)
            self.data = <void*>PyMem_Realloc(self.data, newsize*sizeof(double*))
            self.allocated = newsize
        for i in range(n):
            (<double**>self.data)[size + i] = op[i]
        self.size += n
        
    def extend_double(self, ops):
        for v in ops:
            self._append_double(v)
            
    def as_list_double(self):
        cdef Py_ssize_t i, size = self.size
        cdef list res = []
        
        for i in range(size):
            res.append(self.get_double(i))
        return res
    
    def as_nparray_double(self):
        cdef Py_ssize_t i, size = self.size
        cdef double[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get_double(i)
        return res
    
    def as_memview_double(self):
        cdef Py_ssize_t i, size = self.size
        cdef double[::1] data
        
        res = np.empty(size, 'd')
        data = res
        for i in range(size):
            data[i] = self._get_double(i)
        return data
            
#     def __getitem__(self, index):
#         cdef Py_ssize_t i
#         cdef Py_ssize_t size = self.size
#         # cdef Py_ssize_t start, stop, step
        
#         if PyIndex_Check(<PyObject*>index):
#             i = index
#             if i < 0:
#                 i += size
#             if i < 0 or i >= size:
#                 raise IndexError('%s' % index)
            
#             val = self.data[i]
#             return val
#         # elif PySlice_Check(index):
#         #     if PySlice_GetIndices(index, self.size, &start, &stop, &step) < 0:
#         #         raise IndexError("Invalid slice")
#         #     return self.get_slice(start, stop-start)
#         else:
#             raise IndexError('invalid index: %s' % index)

#     def __setitem__(self, index, val):
#         cdef Py_ssize_t i
#         cdef Py_ssize_t size = self.size
#         # cdef Py_ssize_t start, stop, step

#         if PyIndex_Check(<PyObject*>index):
#             i = index
#             if i < 0:
#                 i += size
#             if i < 0 or i >= size:
#                 raise IndexError('%s' % index)
            
#             self.data[i] = val
#         # elif PySlice_Check(index):
#         #     if PySlice_GetIndices(index, self.size, &start, &stop, &step) < 0:
#         #         raise IndexError("Invalid slice")
#         #     self.set_slice(start, stop-start, val)
#         else:
#             raise IndexError('invalid index: %s' % index)

#     def __delitem__(self, index):
#         cdef Py_ssize_t i = index
#         cdef Py_ssize_t size = self.size
#         cdef double *data = self.data
        
#         if i < 0:
#             i += size
#         if i < 0 or i >= size:
#             raise IndexError('%s' % index)
 
#         size -= 1
#         self.size = size
#         while i < size:
#             data[i] = data[i+1]
#             i += 1
#         data[i] = 0

#         if size + size < self.allocated:
#             newsize = size + (size // 8)
#             self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
#             self.allocated = newsize

#     def __repr__(self):
#         cdef Py_ssize_t i
#         cdef Py_ssize_t size = self.size
#         cdef list temp
        
#         if size == 0:
#             return "list_values([])"
        
#         temp = []
#         for i in range(size):
#             val = self.data[i]
#             temp.append(repr(val))
            
#         return "list_values([" + ", ".join(temp) + "])"
    
#     def __reduce__(self):
#         return self.__class__, (tuple(self),)

#     def append(self, val):
#         cdef Py_ssize_t newsize, size = self.size

#         if size == self.allocated:
#             newsize = resize(size+1)
#             self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
#             self.allocated = newsize
        
#         self.data[self.size] = val
#         self.size += 1

#     def remove(self, ob):
#         cdef Py_ssize_t i, size = self.size
#         cdef double *data = self.data
#         cdef double val, obval = ob

#         i = 0
#         while i < size:
#             if data[i] == obval:
#                 break
#             i += 1
            
#         if i == size:
#             return 

#         size -= 1
#         self.size = size
#         while i < size:
#             data[i] = data[i+1]
#             i += 1
#         data[i] = 0

#     def extend(self, vals):
#         cdef Py_ssize_t i, newsize, size = self.size
#         cdef Py_ssize_t n = len(vals)
#         cdef Py_ssize_t size_n = size + n

#         if size_n > self.allocated:
#             newsize = resize(size_n)

#             self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
#             self.allocated = newsize

#         i = size
#         for val in vals:
#             self.data[i] = val
#             i += 1
#         self.size += n
       
#     def trim(self):
#         if self.size < self.allocated:
#             self.data = <double*>PyMem_Realloc(self.data, self.size*sizeof(double))
#             self.allocated = self.size
            
#     def __len__(self):
#         return self.size
    
#     def __sizeof__(self):
#         return sizeof(list_values) + sizeof(double) * self.allocated
    
#     def __bool__(self):
#         return self.size > 0
    
#     def __iter__(self):
#         return list_values_iter(self)

# cdef class list_values_iter:
#     # cdef list_values op
#     # cdef Py_ssize_t i
    
#     def __init__(self, list_values op):
#         self.op = op
#         self.i = 0
        
#     def __length_hint__(self):
#         return self.op.size
        
#     def __next__(self):
#         if self.i < self.op.size:
#             v = self.op[self.i]
#             self.i += 1
#             return v
#         else:
#             raise StopIteration
            
#     def __iter__(self):
#         return self
        