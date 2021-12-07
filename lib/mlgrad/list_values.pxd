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

# cdef public object sizeof_float
# cdef public object sizeof_pfloat
# cdef public object sizeof_int
# cdef public object sizeof_pint

@cython.no_gc
cdef class list_values:
    cdef Py_ssize_t size
    cdef Py_ssize_t allocated
    cdef void *data
    
    cdef inline float* as_float_array(self)
    cdef inline float** as_pfloat_array(self)

    cdef inline float  _get_float(self, Py_ssize_t i)
    cdef inline float* _get_pfloat(self, Py_ssize_t i)
    cdef inline void _set_float(self, Py_ssize_t i, float p)
    cdef inline void _set_pfloat(self, Py_ssize_t i, float *p)
    
    cdef inline void _append_float(self, float op)
    cdef inline void _append_pfloat(self, float *op)
    
    cdef void _extend_float(self, float *op, Py_ssize_t n)
    cdef void _extend_pfloat(self, float **op, Py_ssize_t n)
    
# cdef class list_values_iter:
#     cdef list_values op
#     cdef Py_ssize_t i
