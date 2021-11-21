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

# from cpython.object cimport PyObject, PyTypeObject
# from cpython.sequence cimport PySequence_Fast, PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE
# from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
# from cpython.mem cimport PyObject_Malloc, PyObject_Realloc, PyObject_Free
# from cpython.slice cimport PySlice_Check, PySlice_GetIndices
# from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_AsDouble

# cdef extern from "Python.h":
#     cdef inline void Py_XDECREF(PyObject*)
#     cdef inline void Py_DECREF(PyObject*)
#     cdef inline void Py_INCREF(PyObject*)
#     cdef inline void Py_XINCREF(PyObject*)
#     cdef inline Py_ssize_t Py_SIZE(object)
#     cdef inline int PyIndex_Check(PyObject*)
#     cdef inline PyTypeObject* Py_TYPE(PyObject*)
#     cdef inline PyObject* PyTuple_GET_ITEM(PyObject*, Py_ssize_t)
    
#     cdef PyTypeObject PyFloat_Type
#     ctypedef struct PyTupleObject:
#         PyObject *ob_item[1]
#     ctypedef struct PyListObject:
#         PyObject **ob_item

cdef inline Py_ssize_t resize(Py_ssize_t size):
    if size < 9:
        return size + (size // 8) + 3
    else:
        return size + (size // 8) + 6

cdef list_double empty_list(Py_ssize_t size, Py_ssize_t itemsize):
    cdef list_double op = <list_double>list_double.__new__(list_double, None)
    
    op.data = <void*>PyMem_Malloc(size*itemsize)
    op.size = op.allocated = size
        
    return <list_double>op

cdef list_double zeros_list_double(Py_ssize_t size):
    cdef list_double op = <list_double>list_double.__new__(list_double, None)
    cdef Py_ssize_t i
    cdef double *p;
    
    op.data = <double*>PyMem_Malloc(size*sizeof(double))
    op.size = op.allocated = size
    p = op.data
    for i in range(size):
        p[i] = 0
        
    return <list_double>op

cdef double* _create_data(Py_ssize_t size, PyObject **args):
    cdef Py_ssize_t i
    cdef double *data
    cdef PyObject *v;

    data = <double*>PyMem_Malloc(size*sizeof(double))
    for i in range(size):
        v = args[i]
        data[i] = PyFloat_AsDouble(<object>v);

    return data

def new_list_double(*args):
    cdef Py_ssize_t i, size = Py_SIZE(args)
    cdef list_double op = <list_double>list_double.__new__(list_double, None)
    cdef double *data;
    cdef PyObject *v;
    
    op.size = op.allocated = size
    data = op.data = <double*>PyMem_Malloc(size*sizeof(double))
    for i in range(size):
        v = PyTuple_GET_ITEM(<PyObject*>args, i)
        if Py_TYPE(v) is &PyFloat_Type:
            data[i] = PyFloat_AS_DOUBLE(<object>v);
        else:
            raise TypeError("This object is not a float")
        
    return <list_double>op

@cython.no_gc
cdef class list_double:
    # cdef Py_ssize_t size
    # cdef Py_ssize_t allocated
    # cdef double *data
    
    def __cinit__(self, args=None):
        cdef Py_ssize_t i, size
        cdef double *data = NULL
        
        if args is None:
            self.size = self.allocated = 0
            self.data = data
            return
        
        size = len(args)
        self.data = data = <double*>PyMem_Malloc(size*sizeof(double))
        for i in range(size):
            v = args[i]
            data[i] = PyFloat_AsDouble(<object>v);                
        self.size = self.allocated = size

    def __dealloc__(self):
        cdef Py_ssize_t i, size = self.size
        cdef double *data = self.data

        for i in range(size):
            data[i] = 0
        PyMem_Free(data)

#     cdef object get_slice(self, Py_ssize_t i, Py_ssize_t n):
#         cdef litelist op = <litelist>make_empty(n)
#         cdef Py_ssize_t j
#         cdef PyObject **op_data = op.data
#         cdef PyObject **data = self.data
        
#         if self.allocated < i + n - 1:
#             raise IndexError('The slice is too large')

#         for j in range(n):
#             op_data[j] = data[i+j]
            
#         return op
            
#     cdef object set_slice(self, Py_ssize_t i, Py_ssize_t n, vals):
#         cdef Py_ssize_t j
#         cdef PyObject *v, *u
#         cdef PyObject **data = self.data
            
#         if self.allocated < i + n - 1:
#             raise IndexError('The slice is too large')

#         tpl = PySequence_Fast(vals, "Invalid arguments")
#         size = Py_SIZE(tpl)
        
#         if n != Py_SIZE(tpl):
#             raise ValueError("incompatible range of indexes")
        
#         for j in range(n):
#             v = PySequence_Fast_GET_ITEM(tpl, j)
#             Py_INCREF(v)
#             u = data[i+j]
#             Py_XDECREF(u)
#             data[i+j] = v
            
    def __getitem__(self, index):
        cdef Py_ssize_t i
        cdef Py_ssize_t size = self.size
        # cdef Py_ssize_t start, stop, step
        
        if PyIndex_Check(<PyObject*>index):
            i = index
            if i < 0:
                i += size
            if i < 0 or i >= size:
                raise IndexError('%s' % index)
            
            val = self.data[i]
            return val
        # elif PySlice_Check(index):
        #     if PySlice_GetIndices(index, self.size, &start, &stop, &step) < 0:
        #         raise IndexError("Invalid slice")
        #     return self.get_slice(start, stop-start)
        else:
            raise IndexError('invalid index: %s' % index)

    def __setitem__(self, index, val):
        cdef Py_ssize_t i
        cdef Py_ssize_t size = self.size
        # cdef Py_ssize_t start, stop, step

        if PyIndex_Check(<PyObject*>index):
            i = index
            if i < 0:
                i += size
            if i < 0 or i >= size:
                raise IndexError('%s' % index)
            
            self.data[i] = val
        # elif PySlice_Check(index):
        #     if PySlice_GetIndices(index, self.size, &start, &stop, &step) < 0:
        #         raise IndexError("Invalid slice")
        #     self.set_slice(start, stop-start, val)
        else:
            raise IndexError('invalid index: %s' % index)

    def __delitem__(self, index):
        cdef Py_ssize_t i = index
        cdef Py_ssize_t size = self.size
        cdef double *data = self.data
        
        if i < 0:
            i += size
        if i < 0 or i >= size:
            raise IndexError('%s' % index)
 
        size -= 1
        self.size = size
        while i < size:
            data[i] = data[i+1]
            i += 1
        data[i] = 0

        if size + size < self.allocated:
            newsize = size + (size // 8)
            self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize

    def __repr__(self):
        cdef Py_ssize_t i
        cdef Py_ssize_t size = self.size
        cdef list temp
        
        if size == 0:
            return "list_double([])"
        
        temp = []
        for i in range(size):
            val = self.data[i]
            temp.append(repr(val))
            
        return "list_double([" + ", ".join(temp) + "])"
    
    def __reduce__(self):
        return self.__class__, (tuple(self),)

    def append(self, val):
        cdef Py_ssize_t newsize, size = self.size

        if size == self.allocated:
            newsize = resize(size+1)
            self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize
        
        self.data[self.size] = val
        self.size += 1

    def remove(self, ob):
        cdef Py_ssize_t i, size = self.size
        cdef double *data = self.data
        cdef double val, obval = ob

        i = 0
        while i < size:
            if data[i] == obval:
                break
            i += 1
            
        if i == size:
            return 

        size -= 1
        self.size = size
        while i < size:
            data[i] = data[i+1]
            i += 1
        data[i] = 0

    def extend(self, vals):
        cdef Py_ssize_t i, newsize, size = self.size
        cdef Py_ssize_t n = len(vals)
        cdef Py_ssize_t size_n = size + n

        if size_n > self.allocated:
            newsize = resize(size_n)

            self.data = <double*>PyMem_Realloc(self.data, newsize*sizeof(double))
            self.allocated = newsize

        i = size
        for val in vals:
            self.data[i] = val
            i += 1
        self.size += n
       
    def trim(self):
        if self.size < self.allocated:
            self.data = <double*>PyMem_Realloc(self.data, self.size*sizeof(double))
            self.allocated = self.size
            
    def __len__(self):
        return self.size
    
    def __sizeof__(self):
        return sizeof(list_double) + sizeof(double) * self.allocated
    
    def __bool__(self):
        return self.size > 0
    
    def __iter__(self):
        return list_double_iter(self)

cdef class list_double_iter:
    # cdef list_double op
    # cdef Py_ssize_t i
    
    def __init__(self, list_double op):
        self.op = op
        self.i = 0
        
    def __length_hint__(self):
        return self.op.size
        
    def __next__(self):
        if self.i < self.op.size:
            v = self.op[self.i]
            self.i += 1
            return v
        else:
            raise StopIteration
            
    def __iter__(self):
        return self
        