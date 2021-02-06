# coding: utf-8

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=False
# cython: initializedcheck=False
# cython: unraisable_tracebacks=True  

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

cimport cython
import numpy as np

from mlgrad.func import func_from_dict

from cython.parallel cimport parallel, prange

from openmp cimport omp_get_num_procs, omp_get_thread_num

cdef int num_procs = omp_get_num_procs()
if num_procs > 4:
    num_procs /= 2
else:
    num_procs = 2

format_double = r"%.2f"
display_precision = 0.005

# cdef double[::1] as_double_array(o):
#     cdef double[::1] ret
#
#     otype = type(o)
#     if otype is list or otype is tuple:
#         ret = np.asarray(o, 'd')
#     else:
#         ret = o
#
#     return ret

cdef inline void fill_memoryview(double[::1] X, double c) nogil:
    cdef int m = X.shape[0]
    memset(&X[0], 0, m*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X) nogil:
    cdef int m = X.shape[0], n = Y.shape[0]

    if n < m:
        m = n
    memcpy(&Y[0], &X[0], m*cython.sizeof(double))
    
cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    cdef int i, j
    cdef int m = X.shape[0], n = X.shape[1]
    memcpy(&Y[0,0], &X[0,0], n*m*cython.sizeof(double))    
    
cdef inline double ident(x):
    return x

def as_array_2d(ob):
    arr = np.asarray(ob, 'd')

    m = len(arr.shape)
    if m == 1:
        return arr.reshape(-1, 1)
    else:
        return arr

def as_array_1d(ob):
    cdef double[::1] arr
    if type(ob) in (list, tuple):
        arr = np.array(ob, 'd')
    elif type(ob) in (int, float):
        arr = np.array([ob], 'd')
    else:
        arr = ob[:]
    return arr

def matdot(A, x, y):
    matrix_dot(A, x, y)

def matdot_t(A, x, y):
    matrix_dot_t(A, x, y)
    
cdef inline double inner_dot(double *a, double *b, Py_ssize_t m) nogil:
    cdef double s
    cdef double a1, a2, a3, a4
    cdef double b1, b2, b3, b4
    
    s = 0
    while m >= 4:
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]

        b1 = b[0]
        b2 = b[1]
        b3 = b[2]
        b4 = b[3]

        s += a1*b1 + a2*b2 + a3*b3 + a4*b4

        m -= 4
        a += 4
        b += 4

    while m > 0:
        s += a[0]*b[0]
        a += 1
        b += 1
        m -= 1

    return s

cdef inline void inner_add(double *b, double *a, double c, Py_ssize_t m) nogil:
    cdef double a1, a2, a3, a4
    
    while m >= 4:
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]

        b[0] += a1 * c
        b[1] += a2 * c
        b[2] += a3 * c
        b[3] += a3 * c

        m -= 4
        a += 4
        b += 4

    while m > 0:
        b[0] += a[0]*c
        a += 1
        b += 1
        m -= 1

cdef inline void inner_assign(double *b, double *a, double c, Py_ssize_t m) nogil:
    cdef double a1, a2, a3, a4
    
    while m >= 4:
        a1 = a[0]
        a2 = a[1]
        a3 = a[2]
        a4 = a[3]

        b[0] = a1 * c
        b[1] = a2 * c
        b[2] = a3 * c
        b[3] = a3 * c

        m -= 4
        a += 4
        b += 4

    while m > 0:
        b[0] = a[0]*c
        a += 1
        b += 1
        m -= 1

cdef class Allocator(object):
    #
    cpdef double[::1] allocate(self, int n):
        return None
    cpdef double[:,::1] allocate2(self, int n, int m):
        return None
    cpdef double[::1] get_allocated(self):
        return None
    cpdef Allocator suballocator(self):
        return self

cdef class ArrayAllocator(Allocator):

    def __init__(self, size):
        self.base = None
        self.size = size
        self.start = 0
        self.allocated = 0
        self.buf = np.zeros((size,), 'd')
    #
    def __repr__(self):
        addr = 0
        if self.base is not None:
            addr = id(self.base)
        return "allocator(%s %s %s %s)" % (addr, self.size, self.start, self.allocated)
    #
    cpdef double[::1] allocate(self, int n):
        cdef double[::1] ar
        cdef ArrayAllocator aa
        
        if n <= 0:
            return None
        
        if self.allocated + n > self.size:
            raise RuntimeError('Memory out of buffer')
        ar = self.buf[self.allocated:self.allocated+n]
        self.allocated += n
        
        aa = self
        while aa.base is not None:
            aa.base.allocated = self.allocated
            aa = aa.base
        return ar
    #
    cpdef double[:,::1] allocate2(self, int n, int m):
        cdef double[:,::1] ar2
        cdef ArrayAllocator aa
        cdef int nm = n*m
        
        if n <= 0 or m <= 0:
            return None
        
        if self.allocated + nm > self.size:
            raise RuntimeError('Memory out of buffer')
        ar = self.buf[self.allocated:self.allocated+nm]
        ar2 = ar.reshape((n,m))
        self.allocated += nm
        
        aa = self
        while aa.base is not None:
            aa.base.allocated = self.allocated
            aa = aa.base
        return ar2
    #
    cpdef double[::1] get_allocated(self):
        return self.buf[self.start:self.allocated]
    #
    cpdef Allocator suballocator(self):
        cdef ArrayAllocator allocator = ArrayAllocator.__new__(ArrayAllocator)

        allocator.buf = self.buf
        allocator.start = self.allocated
        allocator.allocated = self.allocated
        allocator.size = self.size
        allocator.base = self
        return allocator

cdef dict _model_from_dict_table = {}
def register_model(tag):
    def func(cls, tag=tag):
        _model_from_dict_table[tag] = cls
        return cls
    return func

def model_from_dict(ob, init=False):
    tag = ob['name']
    func = _model_from_dict_table[tag]
    mod = func(ob)
    if init:
        mod.allocate()
        mod.init_from(ob)
    return mod

cdef class Model(object):
    #
    def _allocate(self, allocator):
        cdef double[::1] param
        param = allocator.allocate(self.n_param)

        if self.param is not None:
            if param.shape[0] != self.param.shape[0]:
                raise TypeError("The shapes are not equal: %s != %s" % (param.base.shape, self.param.base.shape))
            copy_memoryview(param, self.param)

        self.param = allocator.get_allocated()
        if self.grad is None or self.grad.shape[0] != self.n_param:
            self.grad = np.zeros(self.n_param, 'd')
        if self.grad_x is None or self.grad.shape[0] != self.n_input:
            self.grad_x = np.zeros(self.n_input, 'd')
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate(allocator)
    #
    def init(self, param=None, random=1):
        if self.param is None:
            if param is None:
                if random:
                    self.param = np.random.random(self.n_param)
                else:
                    self.param = np.zeros(self.n_param)
            else:
                self.param = param
        else:
            if param is None:
                if random:
                    self.param[:] = np.random.random(self.n_param)
                else:
                    self.param[:] = np.zeros(self.n_param)
            else:
                self.param[:] = param
    #
    cdef double evaluate(self, const double[::1] X):
        return 0
    #
    def evaluate_all(self, X):
        return [self.evaluate(Xk) for Xk in X]           
    #
    cdef void gradient(self, const double[::1] X, double[::1] grad):
        pass
    #
    cdef void gradient_x(self, const double[::1] X, double[::1] grad):
        pass
    #
    def __call__(self, x):
        cdef double[::1] x1d
        
        x1d = as_array_1d(x)
        return self.evaluate(x1d)
    #
    cpdef Model copy(self, bint share=1):
        return <Model>self
    #

cdef class ConstModel(Model):
    __doc__ = """Constant"""
    #
    def __init__(self, val=0):
        self.param = np.empty(1, 'd')
        self.param[0] = val
        self.grad = np.zeros(1, 'd')
        self.grad_x = np.zeros(1, 'd')
        self.n_param = 1
        self.n_input = 1
    #
    def _allocate(self, allocator):
        pass
    #
    cdef double evaluate(self, const double[::1] X):
        return self.param[0]
    #
    cdef void gradient(self, const double[::1] X, double[::1] grad):
        grad[0] = 1
    #   
    cdef void gradient_x(self, const double[::1] X, double[::1] grad_x):
        grad_x[0] = 0
        
cdef class LinearModel(Model):
    __doc__ = """LinearModel(param)"""
    #
    def __init__(self, o):
        if isinstance(o, (int, long)):
            self.n_input = o
            self.n_param = o + 1
            self.param = None
            self.grad = None
            self.grad_x = None
        else:
            self.param = np.asarray(o, "d")
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_x = np.zeros(self.n_input, 'd')
    #
#     def _allocate(self, allocator):
#         Model._allocate(self, allocator)
#         self.grad = np.zeros(self.n_param, 'd')
#         self.grad_x = np.zeros(self.n_input, 'd')
#     #
    #
    def __reduce__(self):
        return LinearModel, (self.n_input,)
    #
    def __getstate__(self):
        return self.param
    #
    def __setstate__(self, param):
        self.param = param
    #
    def __getnewargs__(self):
        return (self.n_input,)
    #
#     cdef normalize(self):
#         cdef int i, n_param = self.n_param
#         cdef double v, u

#         v = 0
#         for i in range(1, n_param):
#             u = self.param[i]
#             v += u * u
#         v = sqrt(v)
#         for i in range(1, n_param):
#             self.param[i] /= v        
    #
    cdef double evaluate(self, const double[::1] X):
        cdef Py_ssize_t i, n_param = self.n_param
        cdef double v
        cdef double[::1] param = self.param

        #print("LM: %s %s %s" % (self.n_param, tuple(self.param), X))
        v = param[0]
        for i in range(1, n_param):
            v += param[i] * X[i-1]
        return v
    #
    cdef void gradient(self, const double[::1] X, double[::1] grad):
        cdef int i, n_param = self.n_param
        
        grad[0] = 1
        for i in range(1, n_param):
            grad[i] = X[i-1]
    #
    cdef void gradient_x(self, const double[::1] X, double[::1] grad_x):
        cdef int i, n_input = self.n_input
        cdef double[::1] param = self.param

        for i in range(n_input):
            grad_x[i] = param[i+1]
    #
    cpdef Model copy(self, bint share=1):
        cdef LinearModel mod = LinearModel(self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param, 'd')
        mod.grad_x = np.zeros(self.n_input, 'd')
        return <Model>mod
    #
    def _repr_latex_(self):
        if self.param[0]:
            text = format_double % self.param[0]
        else:
            text = ''
        m = self.n_param
        for i in range(1, m):
            par = self.param[i]
            if fabs(par) < display_precision:
                continue
            spar = format_double % par
            if self.param[i] >= 0:
                text += "+%sx_{%s}" % (spar, i)
            else:
                text += "%sx_{%s}" % (spar, i)
        text = "$y(\mathbf{x})=" + text + "$"
        return text
    #
    def as_dict(self):
        return { 'name': 'linear', 
                 'param': (list(self.param) if self.param is not None else None), 
                 'n_input': self.n_input }
    #
    def init_from(self, ob):
        cdef double[::1] param = np.array(ob['param'], 'd')
#         print(param.base)
        copy_memoryview(self.param, param)

@register_model('linear')
def linear_model_from_dict(ob):
    mod = LinearModel(ob['n_input'])
    return mod

cdef class SigmaNeuronModel(Model):
    #
    __doc__ = "Модель сигмоидального нейрона с простыми синапсами"
    #
    def __init__(self, Func outfunc, o):
        self.outfunc = outfunc
        if isinstance(o, (int, long)):
            self.n_param = o + 1
            self.n_input = o
            self.param = None
            self.grad = None
            self.grad_x = None
        else:
            self.param = o
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_x = np.zeros(self.n_input, 'd')
    #
#     def _allocate(self, allocator):
#         Model._allocate(self, allocator)
#         self.grad = np.zeros(self.n_param, 'd')
#         self.grad_x = np.zeros(self.n_input, 'd')
#     #
    cpdef Model copy(self, bint share=1):
        cdef Py_ssize_t n_param = self.n_param
        cdef SigmaNeuronModel mod = SigmaNeuronModel(self.outfunc, self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param, 'd')
        mod.grad_x = np.zeros(self.n_input, 'd')
        return <Model>mod
    #
    cdef double evaluate(self, const double[::1] X):
        cdef int i, n_param = self.n_param
        cdef double s        

        s = self.param[0]
        for i in range(1, n_param):
            s += self.param[i] * X[i-1]
        
        s = self.outfunc.evaluate(s)
        return s
    #
    cdef void gradient(self, const double[::1] X, double[::1] grad):
        cdef int i
        cdef int n_param = self.n_param
        cdef double s, sx
        
        s = self.param[0]
        for i in range(1, n_param):
            s += self.param[i] * X[i-1]

        sx = self.outfunc.derivative(s)

        grad[0] = sx
        for i in range(1, n_param):
            grad[i] = sx * X[i-1]
    #
    cdef void gradient_x(self, const double[::1] X, double[::1] grad_x):
        cdef int i
        cdef int n_input = self.n_input
        cdef int n_param = self.n_param
        cdef double s, sx
                                
        s = self.param[0]
        for i in range(1, n_param):
            s += self.param[i] * X[i-1]

        sx = self.outfunc.derivative(s)            
        for i in range(n_input):
            grad_x[i] = sx * self.param[i+1]
    #
    def as_dict(self):
        return { 'name': 'sigma_neuron', 
                 'func': self.outfunc.to_dict(),
                 'param': (list(self.param) if self.param is not None else None), 
                 'n_input': self.n_input }
    #
    def init_from(self, ob):
        cdef double[::1] param = np.array(ob['param'], 'd')
        copy_memoryview(self.param, param)

@register_model('sigma_neuron')
def sigma_neuron_from_dict(ob):
    mod = SigmaNeuronModel(func_from_dict(ob['func']), ob['n_input'])
    return mod

cdef class ModelLayer:
    
    cdef void forward(self, const double[::1] X):
        pass
    cdef void backward(self, const double[::1] X, double[::1] grad_out, double[::1] grad):
        pass
    cpdef ModelLayer copy(self, bint share=1):
        pass
    
cdef class GeneralModelLayer(ModelLayer):

    def __init__(self, n_input):
        self.n_input = n_input
        self.n_output = 0
        self.n_param = 0
        self.models = []
        self.grad_input = None
        self.output = None
    #
    def _allocate(self, allocator):
        """Allocate mod.param and mod.grad for all models"""
        for mod in self.models:
            n_param = mod.n_param
            if n_param == 0:
                mod.param = None
                continue

            mod._allocate(allocator)
            mod.grad = np.zeros(mod.n_param)
            mod.grad_x = np.zeros(mod.n_input)

        self.param = allocator.get_allocated()
        self.n_param = len(self.param)

        self.n_output = len(self.models)

        self.output = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    cpdef ModelLayer copy(self, bint share=1):
        cdef GeneralModelLayer layer = GeneralModelLayer(self.n_input)
        cdef list models = layer.models
        cdef Model mod

        for mod in self.models:
            models.append(mod.copy(share))

        layer.n_output = self.n_output
        layer.param = self.param
        layer.n_param = self.n_param
        layer.output = np.zeros((self.n_output,), 'd')
        layer.grad_input = np.zeros((self.n_input,), 'd')
        return <ModelLayer>layer
    #
    def append(self, Model mod):
        self.models.append(mod)
    #
    def __getitem__(self, i):
        return self.models[i]
    #
    def __len__(self):
        return len(self.models)
    #
    def __iter__(self):
        return iter(self.models)
    #
    cdef void forward(self, const double[::1] X):
        cdef Model mod
        cdef int j, n_output = self.n_output

        for j in range(n_output):
            mod = <Model>self.models[j]
            self.output[j] = mod.evaluate(X)
    #
    cdef void backward(self, const double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Model mod_j
        cdef int i, j, offset, n_param
        cdef double val_j
        cdef int n_output = self.n_output
        cdef double[::1] grad_in = self.grad_input

        fill_memoryview(grad_in, 0)
        for j in range(n_output):
            val_j = grad_out[j]
            mod_j = <Model>self.models[j]

            mod_j.gradient_x(X, mod_j.grad_x)

            n_param = mod_j.n_param
            offset = j * n_param
            if n_param > 0:
                mod_j.gradient(X, mod_j.grad)
                for i in range(n_param):
                    grad[offset+i] = mod_j.grad[i] * val_j

        for j in range(n_output):
            val_j = grad_out[j]
            mod_j = <Model>self.models[j]
            for i in range(self.n_input):
                grad_in[i] += mod_j.grad_x[i] * val_j                
        #
    def as_dict(self):
        models = []
        for mod in self.models:
            models.append(mod.as_dict())
        return { 'name':'general_model_layer', 'n_input':self.n_input, 'n_output':self.n_output,
                 'models':models}
    #
    def init_from(self, ob):
        for mod, mod_ob in zip(self.mod, ob['models']):
            mod.init_from( mod_ob['param'] )    

@register_model('general_layer')
def general_layer_from_dict(ob):
    layer = GeneralModelLayer(ob['n_input'])
    models = layer.models
    for mod in ob['models']:
        models.append( Model.from_dict(mod) )
    return layer

cdef class SigmaNeuronModelLayer(ModelLayer):

    def __init__(self, Func func, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.n_param = (n_input+1)*n_output
        self.func = func
        self.matrix = None
        self.param = None
        self.grad_input = None
        self.output = None
    #
    def _allocate(self, allocator):
        """Allocate matrix"""
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_output = self.n_output

        self.matrix = allocator.allocate2(n_output, n_input+1)
        self.param = allocator.get_allocated()

        self.output = np.zeros(self.n_output, 'd')
#         self.ss = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
#         self.X = np.zeros((num_procs, self.n_input), 'd')
    #
    cpdef ModelLayer copy(self, bint share=1):
        cdef SigmaNeuronModelLayer layer = SigmaNeuronModelLayer(self.func, self.n_input, self.n_output)
        cdef list models = self.models
        cdef Model mod
        
        layer.matrix = self.matrix
        layer.param = self.param
    
        layer.output = np.zeros(self.n_output, 'd')
        layer.ss = np.zeros(self.n_output, 'd')
        layer.grad_input = np.zeros(self.n_input, 'd')
        self.X = np.zeros((num_procs, self.n_input), 'd')
        return <ModelLayer>layer
    #
    cdef void forward(self, const double[::1] X):
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_output = self.n_output
        cdef Py_ssize_t i, j, size
        cdef double s1, s2, s
        cdef double[:,::1] matrix = self.matrix
        cdef double *output = &self.output[0]
        cdef Func func = self.func
        
        cdef FuncEvaluate func_evaluate = func.evaluate
        
#         size = cython.sizeof(double)*n_input
#         for i in range(num_procs):
#             memcpy(&self.X[i,0], &X[0], size);

#         for j in prange(n_output, nogil=True, num_threads=num_procs):
        for j in range(n_output): 
            s1 = matrix[j,0]
            s2 = inner_dot(&matrix[j,1], &X[0], n_input)
            s = s1 + s2
            if func is None:
                output[j] = s
            else:
                output[j] = func_evaluate(func, s)
    #
    cdef void backward(self, const double[::1] X, const double[::1] grad_out, double[::1] grad):
        cdef Py_ssize_t i, j, jj, offset
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_input1 = n_input + 1
        cdef Py_ssize_t n_output = self.n_output
        cdef double val_j, s, s1, s2, sx
        cdef double* grad_in = &self.grad_input[0]

        cdef double[:,::1] matrix = self.matrix
        cdef double *output = &self.output[0]
        cdef Func func = self.func
        cdef FuncDerivative func_derivative = func.derivative
#         cdef double[::1] ss = self.ss
        
#         fill_memoryview(grad_in, 0) 
        memset(grad_in, 0, n_input*cython.sizeof(double))        
#         for j in prange(n_output, nogil=True, num_threads=num_procs):
        for j in range(n_output):

            s1 = matrix[j,0]
            s2 = inner_dot(&matrix[j,1], &X[0], n_input)
            s = s1 + s2
            
            if func is None:
                sx = grad_out[j]
            else:
                sx = grad_out[j] * func_derivative(func, s)

#         for j in range(n_output):
            inner_add(grad_in, &matrix[j,1], sx, n_input)

#         for j in range(n_output):
#         for j in prange(n_output, nogil=True, num_threads=num_procs):
            jj = j*n_input1
#             sx = ss[j]
            grad[jj] = sx
            
            inner_assign(&grad[jj+1], &X[0], sx, n_input)
    #
    def as_dict(self):
        return { 'name': 'sigma_neuron_layer',
                 'func': self.func.to_dict(),
                 'n_input': self.n_input, 
                 'n_output': self.n_output,
                 'matrix': [list(row) for row in self.matrix]
               }
    #
    #
    def init_from(self, ob):
        cdef double[:,::1] matrix = np.array(ob['matrix'], 'd')
        copy_memoryview2(self.matrix, matrix)

@register_model('sigma_neuron_layer')
def sigma_neuron_layer_from_dict(ob):
    layer = SigmaNeuronModelLayer(ob['n_input'], ob['n_output'])
    return layer
    
cdef class ComplexModel(object):

    cdef void forward(self, const double[::1] X):
        pass
    #
    cdef void backward(self, const double[::1] X, double[::1] grad_u, double[::1] grad):
        pass
    #
    
cdef class MLModel(ComplexModel):

    cpdef MLModel copy(self, bint share=1):
        pass
    #
    def __call__(self, x):
        cdef double[::1] x1d
        
        x1d = as_array_1d(x)
        #print('__call__', x1d)
        self.forward(x1d)
        return self.output.copy()
    #

cdef class FFNetworkModel(MLModel):

    def __init__(self):
        self.n_param = 0
        self.layers = []
        self.param = None
    #
    def add(self, layer):
        n_layer = len(self.layers)
        if n_layer == 0:
            self.n_input = layer.n_input
        else:
            n_output = self.layers[n_layer-1].n_output
            if n_output != layer.n_input:
                raise RuntimeError("Previous layer n_output=%s, layer n_input=%s" % (n_output, layer.n_input))
        self.layers.append(layer)
        self.n_param += layer.n_param 
        self.n_output = layer.n_output
    #
    def __getitem__(self, i):
        return self.layers[i]
    #
    def __len__(self):
        return len(self.layers)
    #
    def __iter__(self):
        return iter(self.layers)
    #
    def evaluate_all(self, X):
        return [self(x) for x in X]
    #    
    def _allocate(self, allocator):
        """Allocate mod.param and mod.grad for all models"""
        
        for layer in self.layers:
            if layer.n_param > 0:
                layer_allocator = allocator.suballocator()
                layer._allocate(layer_allocator)

        self.param = allocator.get_allocated()

        n_layer = len(self.layers)
        layer = self.layers[n_layer-1]
        self.output = layer.output
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate(allocator)
    #
    cpdef MLModel copy(self, bint share=1):
        cdef FFNetworkModel ml = FFNetworkModel()
        cdef ModelLayer layer
        cdef int n_layer
        
        ml.param = self.param
        ml.n_param = self.n_param
        ml.n_input = self.n_input
        ml.n_output = self.n_output
        for layer in self.layers:
            ml.layers.append(layer.copy(share))
        
        n_layer = len(ml.layers)
        layer = ml.layers[n_layer-1]
        ml.output = layer.output
            
        return <MLModel>ml
    #
    cdef void forward(self, const double[::1] X):
        cdef int i, n_layer
        cdef ModelLayer layer
        cdef const double[::1] input, output
        cdef list layers = self.layers

        n_layer = len(self.layers)
        input = X
        for i in range(n_layer):
            layer = <ModelLayer>layers[i]
            layer.forward(input)
            input = layer.output
#         self.output = layer.output

    cdef void backward(self, const double[::1] X, double[::1] grad_u, double[::1] grad):
        cdef int n_layer = PyList_GET_SIZE(<PyObject*>self.layers)
        cdef int j, l, m, m0
        cdef ModelLayer layer, prev_layer
#         cdef double[::1] grad_in
        cdef const double[::1] grad_out, input
        cdef list layers = self.layers

        m = len(grad)
        l = n_layer-1
        grad_out = grad_u
        while l >= 0:
            layer = <ModelLayer>layers[l]
            if l > 0:
                prev_layer = <ModelLayer>layers[l-1]
                input = prev_layer.output
            else:
                input = X
            m0 = m - layer.n_param
            if layer.n_param > 0:
                layer.backward(input, grad_out, grad[m0:m])
            else:
                layer.backward(input, grad_out, None)
            grad_out = layer.grad_input
            l -= 1
            m = m0
    #
    def as_dict(self):
        layers = []
        for layer in self.layers:
            layers.append(layer.as_json())
        return {'name':'ff_nn', 'n_input':self.n_input, 'layers':layers}
    #
    def init_from(self, ob):
        for layer, layer_ob in zip(self.layers, ob['layers']):
            layer.init_from( layer_ob['param'] )    
        

@register_model('ff_nn')
def ff_ml_network_from_dict(ob):
    nn = FFNetworkModel()
    for layer in ob['layers']:
        nn.add( Model.from_dict(layer) )
    return nn

cdef class FFNetworkFuncModel(Model):
    #
    def __init__(self, head, body):
        self.body = body
        self.head = head
        self.n_param = self.head.n_param + self.body.n_param
        self.n_input = self.body.n_input
        self.grad = None
        self.grad_x = None
    #
    def _allocate(self, allocator):
        #print("NN", allocator)
        if self.head.n_param > 0:
            self.head._allocate(allocator)

        body_allocator = allocator.suballocator()
        
        self.body._allocate(body_allocator)
        
        self.param = allocator.get_allocated()
        self.n_param = len(self.param)
        #print("NN", allocator)
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate(allocator)
    #
    cpdef Model copy(self, bint share=1):
        cdef FFNetworkFuncModel mod = FFNetworkFuncModel(self.head.copy(share), self.body.copy(share))
        
        mod.param = self.param
        
        return <Model>mod
    #
    cdef double evaluate(self, const double[::1] X):
        self.body.forward(X)
        return self.head.evaluate(self.body.output)
    #
    cdef void gradient(self, const double[::1] X, double[::1] grad):
        cdef int i, j, n
        
        if self.head.n_param > 0:
            self.head.gradient(self.body.output, grad[:self.head.n_param])
        self.head.gradient_x(self.body.output, self.head.grad_x)
        self.body.backward(X, self.head.grad_x, grad[self.head.n_param:])
    #
    def as_dict(self):
        d = {}
        d['body'] = self.body.as_json()
        d['head'] = self.head.as_json()
        return d
    #
    def init_from(self, ob):
        self.head.init_from(ob['head'])
        self.body.init_from(ob['body'])
    #

@register_model('ff_nn_func')
def ff_ml_network_func_from_dict(ob):
    head = Model.from_dict(ob['head'])
    body = Model.from_dict(ob['body'])
    nn = FFNetworkFuncModel(head, body)
    return nn
    
# cdef class Polynomial(Model):
#     #
#     def __init__(self, param):
#         self.param = np.asarray(param, 'd')
#     #
#     cdef double evaluate(self, const double[::1] X):
#         cdef double *param_ptr = &self.param[0]
#         cdef double x = X[0]
#         cdef double val = 0
#         cdef int i, m = self.param.shape[0]
        
#         i = m-1
#         while i >= 0:
#             val = val * x + param_ptr[i]
#             i -= 1
#         return val
        
#     cdef gradient(self, const double[::1] X, double[::1] grad):
#         cdef double x = X[0]
#         cdef double val = 1.0
#         cdef int i, m = self.param.shape[0]
        
#         for i in range(m):
#             grad[i] = val
#             val *= x

#     cdef gradient_x(self, const double[::1] X, double[::1] grad):
#         cdef double x = X[0]
#         cdef double val = 1.0
#         cdef int i, m = self.param.shape[0]
        
#         for i in range(1, m):
#             grad[i] = val * i 
#             val *= x

cdef class SquaredModel(Model):
    #
    def __init__(self, mat):
        cdef double[:,::1] matrix
        
        _mat = np.asarray(mat)
        _par = _mat.reshape(-1)
        self.matrix = _mat
        self.param = _par
        #self.matrix_grad = np.zeros_like(_mat)
        
        self.n_param = len(_par)
        self.n_input = _mat.shape[1] - 1
        
        self.grad = np.zeros(self.n_param, 'd')
        self.grad_x = np.zeros(self.n_input, 'd')
    #
    cdef double evaluate(self, const double[::1] X):
        cdef double val, s
        #cdef double[:,::1] mat = self.matrix
        cdef int i, j, n, m
        
        n = self.matrix.shape[0]
        m = self.matrix.shape[1]
        val = 0
        for j in range(n):
            s = self.matrix[j,0]
            for i in range(1,m):
                s += self.matrix[j,i] * X[i-1]
            val += s*s
        return val
    #
    cdef void gradient_x(self, const double[::1] X, double[::1] y):
        cdef double val, s
        #cdef double[:,::1] mat = self.matrix
        cdef int i, j, n, m
        
        n = self.matrix.shape[0]
        m = self.matrix.shape[1]

        s = 0
        for i in range(m):
            s = self.matrix[j,0]
            for j in range(n):
                s += self.matrix[j,i]
            s *= 2
            #s *= mat[j,]
    #
    cdef void gradient(self, const double[::1] X, double[::1] y):
        cdef double val, s
        #cdef double[:,::1] mat = self.matrix
        cdef int i, j, n, m, k
        
        n = self.matrix.shape[0]
        m = self.matrix.shape[1]
        
        k = 0
        for j in range(n):
            s = self.matrix[j,0]
            for i in range(1,m):
                s += self.matrix[j,i] * X[i-1]
            s *= 2
            
            y[k] = s
            k += 1
            for i in range(1, m):
                y[k] = s * X[i-1]
                k += 1
                
        
            
        
            
# cdef class WinnerModel(Model):
#     #
#     def __init__(self, Func outfunc, n_input):
#         self.outfunc = func
#         self.n_param = 0
#         self.n_input = n_input
#         self.param = None
#     #
#     cdef double evaluate(self, const double[::1] X):
#         cdef i, n = X.shape[0]
#         cdef int i_max = 0
#         cdef double x, x_max = X[0]
#
#         for i in range(n):
#             x = X[i]
#             if x > x_max:
#                 i_max = i
#         return self.outfunc.evaluate(X[i_max])
#     #
#     cdef void gradient(self, const double[::1] X, double[::1] grad):
#         pass
#     #
#     cdef void gradient_x(self, const double[::1] X, double[::1] grad):
#         cdef i, n = X.shape[0]
#         cdef int i_max = 0
#         cdef double x, x_max = X[0]
#
#         fill_memoryview(grad, 0.)
#         for i in range(n):
#             x = X[i]
#             if x > x_max:
#                 i_max = i
#
#         grad[i_max] = self.outfunc.derivative(X[i_max])
            
# def gradient(obj, X, Y):
#     cdef Model mod = obj
#     return mod.gradient(as_array_1d(X), as_array_1d(Y))

# cdef class TrainingModel(object):
#     pass
#
# cdef class TrainingSModel(TrainingModel):
#     #
#     def __init__(self, Model model, Loss loss_func):
#         self.model = mod
#         self.loss_func = loss_func
#     #
#     def gradient(self, const double[::1] X, double y):
#         cdef double yk
#
#         yk = self.model.evaluate(X)
#         lval = self.loss_func.evaluate(y, yk)
#         lval_deriv =
#

