# coding: utf-8

# cython: language_level=3

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

from cpython.object cimport PyObject, PyTypeObject
from cpython.sequence cimport PySequence_Fast, PySequence_Fast_GET_ITEM, PySequence_Fast_GET_SIZE
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from cpython.mem cimport PyObject_Malloc, PyObject_Realloc, PyObject_Free
from cpython.slice cimport PySlice_Check, PySlice_GetIndices
from cpython.float cimport PyFloat_AS_DOUBLE, PyFloat_AsDouble

cdef extern from "Python.h":
    cdef void Py_XDECREF(PyObject*)
    cdef void Py_DECREF(PyObject*)
    cdef void Py_INCREF(PyObject*)
    cdef void Py_XINCREF(PyObject*)
    cdef Py_ssize_t Py_SIZE(object)
    cdef int PyIndex_Check(PyObject*)
    cdef PyTypeObject* Py_TYPE(PyObject*)
    cdef PyObject* PyTuple_GET_ITEM(PyObject*, Py_ssize_t)
    
    cdef PyTypeObject PyFloat_Type
    ctypedef struct PyTupleObject:
        PyObject *ob_item[1]
    ctypedef struct PyListObject:
        PyObject **ob_item

# cdef list_values empty_list_values(Py_ssize_t size)
# cdef list_values zeros_list_values(Py_ssize_t size)

# cdef public object sizeof_double
# cdef public object sizeof_pdouble
# cdef public object sizeof_int
# cdef public object sizeof_pint

@cython.no_gc
cdef class list_double:
    cdef Py_ssize_t size
    cdef Py_ssize_t allocated
    cdef double *data

    cdef inline double  _get(self, Py_ssize_t i)
    cdef inline void _set(self, Py_ssize_t i, double p)
    
    cdef void _append(self, double op)
    cdef void _extend(self, double *op, Py_ssize_t n)

@cython.no_gc
cdef class list_int:
    cdef Py_ssize_t size
    cdef Py_ssize_t allocated
    cdef int *data
    
    cdef inline int  _get(self, Py_ssize_t i)
    cdef inline void _set(self, Py_ssize_t i, int p)
    
    cdef void _append(self, int op)
    cdef void _extend(self, int *op, Py_ssize_t n)
