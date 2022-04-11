# coding: utf-8 

# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: initializedcheck=False
# cython: unraisable_tracebacks=True  

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

from mlgrad.func import func_from_dict
from mlgrad.func import func_from_dict

# from cython.parallel cimport parallel, prange

# cimport mlgrad.inventory as inventory

format_double = r"%.2f"
display_precision = 0.005

cdef inline void fill_memoryview(double[::1] X, double c) nogil:
    memset(&X[0], 0, X.shape[0]*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X) nogil:
    memcpy(&Y[0], &X[0], X.shape[0]*cython.sizeof(double))
    
cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    memcpy(&Y[0,0], &X[0,0], X.shape[0]*X.shape[1]*cython.sizeof(double))    
    
cdef inline double ident(x):
    return x

def as_array2d(ob):
    arr = np.asarray(ob, 'd')

    m = len(arr.shape)
    if m == 1:
        return arr.reshape(-1, 1)
    elif m > 2:
        raise TypeError('number of axes > 2!')
    else:
        return arr

def as_array1d(ob):
    arr = np.asarray(ob, 'd')
    if len(arr.shape) > 1:
        arr = arr.ravel()
    return arr

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
        self.buf = np.zeros(size, 'd')
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
        ar2 = ar.reshape(n,m)
        self.allocated += nm
        
        aa = self
        while aa.base is not None:
            aa.base.allocated = self.allocated
            aa = aa.base
        return ar2
    #
    cpdef double[::1] get_allocated(self):
        self.buf[self.start:self.allocated] = 0
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

cdef class BaseModel:
    #
    cdef double _evaluate(self, double[::1] X):
        return 0
    #
    def evaluate_all(self, X):
        N = len(X)
        Y = np.empty(N, 'd')
        self._evaluate_all(X, Y)
        return Y
    #
    cdef _evaluate_all(self, double[:,::1] X, double[::1] Y):
        cdef Py_ssize_t k, N = X.shape[0]
        
        for k in range(N):
            Y[k] = self._evaluate(X[k])
    #
    cpdef copy(self, bint share=0):
        return self

cdef class Model(BaseModel):
    #
    def _allocate(self, allocator):
        cdef double[::1] param
        param = allocator.allocate(self.n_param)

        if self.param is not None:
            if param.shape[0] < self.param.shape[0]:
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
    cpdef init_param(self, param=None, bint random=1):
        cdef double[::1] r 
        if self.param is None:
            if param is None:
                if random:
                    self.param = np.random.random(self.n_param)
                else:
                    self.param = np.zeros(self.n_param, 'd')
            else:
                self.param = param
        else:
            if param is None:
                if random:
                    r = np.random.random(self.n_param)
                    inventory._move(&self.param[0], &r[0], self.n_param) 
                else:
                    inventory._fill(&self.param[0], 0, self.n_param)
            else:
                self.param[:] = param
        # print(np.array(self.param, 'd'), self.n_param)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        pass
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad):
        pass
    #
    def __call__(self, x):
        cdef double[::1] x1d = as_array1d(x)        
        return self._evaluate(x1d)
    #
    cpdef Model copy(self, bint share=1):
        return <Model>self

# cdef class ConstModel(Model):
#     __doc__ = """Constant"""
#     #
#     def __init__(self, val=0):
#         self.param = np.empty(1, 'd')
#         self.param[0] = val
#         self.grad = np.zeros(1, 'd')
#         self.grad_x = np.zeros(1, 'd')
#         self.n_param = 1
#         self.n_input = 1
#     #
#     def _allocate(self, allocator):
#         pass
#     #
#     cdef double _evaluate(self, double[::1] X):
#         return self.param[0]
#     #
#     cdef void _gradient(self, double[::1] X, double[::1] grad):
#         grad[0] = 1
#     #   
#     cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
#         grad_x[0] = 0
        
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
            self.param = np.asarray(o, 'd')
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
    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, n = self.n_input
        cdef double v
        cdef double[::1] param = self.param

        v = param[0]
        for i in range(self.n_input):
            v += param[i+1] * X[i]
        return v
        # cdef double *param = &self.param[0]
        # return param[0] + inventory._conv(&X[0], &param[1], self.n_input)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i
        
        grad[0] = 1.
        # inventory._move(&grad[1], &X[0], self.n_input)
        for i in range(self.n_input):
            grad[i+1] = X[i]
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        cdef Py_ssize_t i
        cdef double[::1] param = self.param

        # inventory._move(&grad_x[0], &self.param[1], self.n_input)
        for i in range(self.n_input):
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
    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i
        cdef double[::1] param = self.param
        # cdef double *Xp = &X[0]
        cdef double s

        s = param[0]
        for i in range(self.n_input):
            s += param[i+1] * X[i]
        
        s = self.outfunc._evaluate(s)
        return s
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i
        cdef double[::1] param = self.param
        # cdef double *Xp = &X[0]
        # cdef double *G = &grad[1]
        cdef double s, sx
        
        s = param[0]
        for i in range(self.n_input):
            s += param[i+1] * X[i]

        sx = self.outfunc._derivative(s)

        grad[0] = sx
        for i in range(self.n_input):
            grad[i+1] = sx * X[i]
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        cdef Py_ssize_t i
        cdef Py_ssize_t n_input = self.n_input
        cdef double s, sx
        cdef double[::1] param = self.param
        # cdef double *Xp = &X[0]
        # cdef double *G = &grad_x[0]
                                
        s = param[0]
        for i in range(n_input):
            s += param[i+1] * X[i]

        sx = self.outfunc._derivative(s)

        for i in range(n_input):
            grad_x[i] = sx * param[i+1]
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
    
    cdef void forward(self, double[::1] X):
        pass
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        pass
    cpdef ModelLayer copy(self, bint share=1):
        pass
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
    #
    
cdef class GeneralModelLayer(ModelLayer):
    #
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

            mod_allocator = allocator.suballocator()
            mod._allocate(mod_allocator)

        param = allocator.get_allocated()

        n_param0 = len(self.param)
        n_param1 = len(param)
        if n_param0 > 0:
            if n_param0 <= n_param1:
                for i in range(n_param0):
                    param[i] = self.param[i]
        self.param = param
        self.n_param = len(param)

        self.n_output = len(self.models)

        self.output = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
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
        if self.n_input != mod.n_input:
            raise ValueError("layer.n_input: %s != model.n_input: %s" % (self.n_input, mod.n_input))
        self.models.append(mod)
        self.n_param += mod.n_param
        self.n_output += 1
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
    cdef void forward(self, double[::1] X):
        cdef Model mod
        cdef Py_ssize_t j, n_output = self.n_output

        for j in range(n_output):
            mod = <Model>self.models[j]
            self.output[j] = mod._evaluate(X)
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Model mod_j
        cdef Py_ssize_t i, j, n_param
        cdef double val_j
        cdef Py_ssize_t n_output = self.n_output
        cdef double[::1] grad_in = self.grad_input
        cdef double *G = &grad[0]

        fill_memoryview(grad_in, 0)
        for j in range(n_output):
            val_j = grad_out[j]
            mod_j = <Model>self.models[j]

            n_param = mod_j.n_param
            if n_param > 0:
                mod_j._gradient(X, mod_j.grad)
                for i in range(n_param):
                    G[i] = mod_j.grad[i] * val_j
                G += n_param

            mod_j._gradient_x(X, mod_j.grad_x)
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
        models.append( model_from_dict(mod) )
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
        self.ss = None
        # self.first_time = 1
    #
    def init_param(self):
        if self.n_param > 0 and self.param is not None:
            self.param = np.random.random(self.n_param)
    #
    def _allocate(self, allocator):
        """Allocate matrix"""
        # cdef Py_ssize_t n_input = self.n_input
        # cdef Py_ssize_t n_output = self.n_output

        self.matrix = allocator.allocate2(self.n_output, self.n_input+1)
        self.param = allocator.get_allocated()

        self.output = np.zeros(self.n_output, 'd')
        self.ss = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
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
        return <ModelLayer>layer
    #
    cdef void forward(self, double[::1] X):
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_output = self.n_output
        cdef Py_ssize_t i, j
        cdef double s
        cdef double[:,::1] matrix = self.matrix
        cdef double[::1] output = self.output
        cdef double[::1] ss = self.ss
        cdef Func func = self.func
         
        for j in range(n_output):
            s = matrix[j,0]
            for i in range(n_input):
                s += matrix[j, i+1] * X[i]
            ss[j] = s

        if func is not None:
            for j in range(n_output):
                output[j] = func._evaluate(ss[j])
        else:
            for j in range(n_output):
                output[j] = ss[j]
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Py_ssize_t i, j, jj
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_input1 = n_input + 1
        cdef Py_ssize_t n_output = self.n_output
        cdef double val_j, s, sx
        cdef double[::1] grad_in = self.grad_input

        cdef double[:,::1] matrix = self.matrix
        cdef double[::1] output = self.output
        cdef double[::1] ss = self.ss
        cdef Func func = self.func
        cdef bint is_func = (func is not None)
        
        # fill_memoryview(grad_in, 0)
        inventory.fill(grad_in, 0)
        for j in range(n_output):
            s = matrix[j,0]
            for i in range(n_input):
                s += matrix[j,i+1] * X[i]
            ss[j] = s

        if is_func:
            for j in range(n_output):            
                ss[j] = grad_out[j] * func._derivative(ss[j])
        else:
            for j in range(n_output):            
                ss[j] = grad_out[j]

        for j in range(n_output):            
            sx = ss[j]
            for i in range(n_input):
                grad_in[i] += sx * matrix[j,i+1]

        jj = 0
        for j in range(n_output):            
            grad[jj] = sx = ss[j]
            # grad[jj] = sx
            jj += 1
            for i in range(n_input):
                grad[jj] = sx * X[i]
                jj += 1
    #
    def as_dict(self):
        return { 'name': 'sigma_neuron_layer',
                 'func': self.func.to_dict(),
                 'n_input': self.n_input, 
                 'n_output': self.n_output,
                 'matrix': [list(row) for row in self.matrix]
               }
    #
    def init_from(self, ob):
        cdef double[:,::1] matrix = np.array(ob['matrix'], 'd')
        copy_memoryview2(self.matrix, matrix)

@register_model('sigma_neuron_layer')
def sigma_neuron_layer_from_dict(ob):
    layer = SigmaNeuronModelLayer(ob['n_input'], ob['n_output'])
    return layer

cdef class LinearFuncModel(BaseModel):
    #
    def __init__(self):
        self.models = []
        self.weights = list_doubles(0)
    #
    def add(self, Model mod, weight=1.0):
        # if mod.n_input != self.n_input:
        #     raise TypeError('model.n_input != self.n_input')
        self.models.append(mod)
        self.weights.append(weight)
    #
    cpdef copy(self, bint share=1):
        cdef LinearFuncModel mod = LinearFuncModel()
        mod.models = self.models[:]
        mod.weights = self.weights.copy()
        return mod
    #
    cdef double _evaluate(self, double[::1] X):
        cdef double w, s
        cdef Model mod
        cdef list models = self.models
        cdef Py_ssize_t j, m=self.weights.size
        cdef double *weights = self.weights.data

        s = 0
        for j in range(m):
            mod = <Model>models[j]
            # w = weights[j]
            s += weights[j] * mod._evaluate(X)
        return s
    #
    def evaluate_all(self, double[:,::1] X):
        cdef Py_ssize_t k, N = len(X)
        
        Y = np.empty(N, 'd')
        for k in range(N):
            Y[k] = self._evaluate(X[k])
        return Y
    #
    def __call__(self, X):
        cdef double[::1] XX = X
        return self._evaluate(XX)
    #
    # def as_dict(self):
    #     d = {}
    #     d['body'] = self.body.as_json()
    #     d['head'] = self.head.as_json()
    #     return d
    #
    # def init_from(self, ob):
    #     self.head.init_from(ob['head'])
    #     self.body.init_from(ob['body'])
    #
    
cdef class MLModel:

    cdef void forward(self, double[::1] X):
        pass
    #
    cdef void backward(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        pass
    #
    cdef void backward2(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        self.forward(X)
        self.backward(X, grad_u, grad)
    #
    
    cpdef MLModel copy(self, bint share=1):
        pass
    #
    def init_param(self):
        # self.param = np.random.random(self.n_param)
        for layer in self.layers:
            layer.init_param()
    #
    def __call__(self, x):
        cdef double[::1] x1d = as_array1d(x)
        
        self.forward(x1d)
        return self.output.copy()
    #

cdef class FFNetworkModel(MLModel):

    def __init__(self):
        self.n_param = 0
        self.layers = []
        self.param = None
        self.is_forward = 0
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
        # self.output = layer.output
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
        cdef Py_ssize_t n_layer
        
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
    cdef void forward(self, double[::1] X):
        cdef Py_ssize_t i, n_layer
        cdef ModelLayer layer
        cdef double[::1] input, output
        cdef list layers = self.layers

        n_layer = len(self.layers)
        input = X
        for i in range(n_layer):
            layer = <ModelLayer>layers[i]
            layer.forward(input)
            input = layer.output
#         self.output = layer.output
        self.is_forward = 1

    cdef void backward(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        cdef Py_ssize_t n_layer = PyList_GET_SIZE(<PyObject*>self.layers)
        cdef Py_ssize_t j, l, m, m0
        cdef ModelLayer layer, prev_layer
#         cdef double[::1] grad_in
        cdef double[::1] grad_out, input
        cdef list layers = self.layers

        if not self.is_forward:
            self.forward(X)
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
        self.is_forward = 0
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
    cdef double _evaluate(self, double[::1] X):
        self.body.forward(X)
        return self.head._evaluate(self.body.output)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef int i, j, n
        cdef Model head = self.head
        cdef MLModel body = self.body
        
        # body.forward(X) 
        if head.n_param > 0:
            head._gradient(body.output, grad[:head.n_param])
        head._gradient_x(body.output, head.grad_x)
        body.backward(X, head.grad_x, grad[head.n_param:])
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
#     cdef double _evaluate(self, double[::1] X):
#         cdef double *param_ptr = &self.param[0]
#         cdef double x = X[0]
#         cdef double val = 0
#         cdef int i, m = self.param.shape[0]
        
#         i = m-1
#         while i >= 0:
#             val = val * x + param_ptr[i]
#             i -= 1
#         return val
        
#     cdef _gradient(self, double[::1] X, double[::1] grad):
#         cdef double x = X[0]
#         cdef double val = 1.0
#         cdef int i, m = self.param.shape[0]
        
#         for i in range(m):
#             grad[i] = val
#             val *= x

#     cdef _gradient_x(self, double[::1] X, double[::1] grad):
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
    cdef double _evaluate(self, double[::1] X):
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
    cdef void _gradient_x(self, double[::1] X, double[::1] y):
        cdef double val, s
        #cdef double[:,::1] mat = self.matrix
        cdef int i, j, n, m
        
        n = self.matrix.shape[0]
        m = self.matrix.shape[1]

        s = 0
        for j in range(m):
            s = self.matrix[j,0]
            for i in range(n):
                s += self.matrix[j,i]
            s *= 2
            #s *= mat[j,]
    #
    cdef void _gradient(self, double[::1] X, double[::1] y):
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
#     cdef double _evaluate(self, double[::1] X):
#         cdef i, n = X.shape[0]
#         cdef int i_max = 0
#         cdef double x, x_max = X[0]
#
#         for i in range(n):
#             x = X[i]
#             if x > x_max:
#                 i_max = i
#         return self.outfunc._evaluate(X[i_max])
#     #
#     cdef void _gradient(self, double[::1] X, double[::1] grad):
#         pass
#     #
#     cdef void _gradient_x(self, double[::1] X, double[::1] grad):
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
#         grad[i_max] = self.outfunc._derivative(X[i_max])
            
# def _gradient(obj, X, Y):
#     cdef Model mod = obj
#     return mod._gradient(as_array_1d(X), as_array_1d(Y))

# cdef class TrainingModel(object):
#     pass
#
# cdef class TrainingSModel(TrainingModel):
#     #
#     def __init__(self, Model model, Loss loss_func):
#         self.model = mod
#         self.loss_func = loss_func
#     #
#     def _gradient(self, double[::1] X, double y):
#         cdef double yk
#
#         yk = self.model._evaluate(X)
#         lval = self.loss_func._evaluate(y, yk)
#         lval_deriv =
#

