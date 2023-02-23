# coding: utf-8 

# The MIT License (MIT)
#
# Copyright (c) <2015-2022> <Shibzukhov Zaur, szport at gmail dot com>
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

# from cython.parallel cimport parallel, prange

cimport mlgrad.inventory as inventory

format_double = r"%.2f"
display_precision = 0.005

cdef inline void fill_memoryview(double[::1] X, double c) nogil:
    memset(&X[0], 0, X.shape[0]*cython.sizeof(double))    

cdef inline void copy_memoryview(double[::1] Y, double[::1] X) nogil:
    memcpy(&Y[0], &X[0], <Py_ssize_t>X.shape[0]*cython.sizeof(double))
    
cdef inline void copy_memoryview2(double[:,::1] Y, double[:,::1] X):
    memcpy(&Y[0,0], &X[0,0], <Py_ssize_t>X.shape[0]*<Py_ssize_t>X.shape[1]*cython.sizeof(double))    
    
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
        mod.allocate_param()
        mod.init_from(ob)
    return mod

cdef class BaseModel:
    #
    cdef double _evaluate(self, double[::1] X):
        return 0
    #
    def evaluate_all(self, X):
        Y = np.empty(len(X), 'd')
        self._evaluate_all(X, Y)
        return Y
    #
    def evaluate(self, X):
        return self._evaluate(X)
    #
    cdef _evaluate_all(self, double[:,::1] X, double[::1] Y):
        cdef Py_ssize_t k, N = <Py_ssize_t>X.shape[0]
        
        for k in range(N):
            Y[k] = self._evaluate(X[k])
    #
    cpdef copy(self, bint share=0):
        return self

cdef class Model(BaseModel):
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        if self.param is not None:
            raise RuntimeError('param is allocated already ')
        self._allocate_param(allocator)
    #
    def _allocate_param(self, allocator):
        """
        Распределение памяти под `self.param` (если `self.param` уже создан, то его содержимое 
        переносится во вновь распределнное пространство)
        
        """        
        self.ob_param = allocator.allocate(self.n_param)
        if self.param is not None:
            with cython.boundscheck(True):
                self.ob_param[:] = self.param

        self.param = self.ob_param
        self.grad = np.zeros(self.n_param, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    def init_param(self, param=None, random=1):
        if random and param is None:
            r = np.random.random(self.n_param)
        elif param is None:
            r = np.zeros(self.n_param, 'd')
        else:
            if len(param) != self.n_param:
                raise TypeError(f"len(param) = {len(param)} n_param = {self.n_param}")
            r = param

        if self.param is None:
            self.ob_param = self.param = r
        else:
            self.ob_param[:] = r
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        pass
    #
    def gradient(self, X):
        grad = np.empty(self.n_param, 'd')
        self._gradient(X, grad)
        return grad
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        pass
    #
    def gradient_input(self, X):
        grad_input = np.empty(self.n_input, 'd')
        self._gradient_input(X, grad_input)
        return grad_input
    #
    def evaluate(self, x):
        cdef double[::1] x1d = as_array1d(x)        
        return self._evaluate(x1d)
    #
    cpdef Model copy(self, bint share=1):
        return <Model>self
    #
    # cdef update_param(self, double[::1] param):
    #     inventory.sub(self.param, param)
    
cdef class ModelView(Model):
    #
    def __init__(self, Model model):
        self.model = model
    #
    cdef double _evaluate(self, double[::1] X):
        return self.model._evaluate(X)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        self.model._gradient(X, grad)
        
cdef class LinearModel(Model):
    __doc__ = """LinearModel(param)"""
    #
    def __init__(self, o):
        if type(o) == type(1):
            self.n_input = o
            self.n_param = o + 1
            self.param = self.ob_param = None
            self.grad = None
            self.grad_input = None
            self.param = self.ob_param = None
            # self.is_allocated = 0
        else:
            self.param = self.ob_param = np.asarray(o, 'd')
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_input = np.zeros(self.n_input, 'd')
            # self.is_allocated = 1
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
        # cdef Py_ssize_t i, n = self.n_input
        # cdef double v
        # cdef double[::1] param = self.param

        # v = param[0]
        # for i in range(self.n_input):
        #     v += param[i+1] * X[i]
        # return v
        cdef double *param = &self.param[0]
        return param[0] + inventory._conv(&X[0], &param[1], self.n_input)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        # cdef Py_ssize_t i
        
        grad[0] = 1.
        inventory._move(&grad[1], &X[0], self.n_input)
        # for i in range(self.n_input):
        #     grad[i+1] = X[i]
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        # cdef Py_ssize_t i
        # cdef double[::1] param = self.param

        inventory._move(&grad_input[0], &self.param[1], self.n_input)
        # for i in range(self.n_input):
        #     grad_input[i] = param[i+1]
    #
    cpdef Model copy(self, bint share=1):
        cdef LinearModel mod = LinearModel(self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param, 'd')
        mod.grad_input = np.zeros(self.n_input, 'd')
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
        if self.param is None:
            self.param = param
        else:
            self.param[:] = param
        self.n_param = <Py_ssize_t>param.shape[0]
        self.n_input = self.n_param - 1

@register_model('linear')
def linear_model_from_dict(ob):
    mod = LinearModel(ob['n_input'])
    mod.init_from(ob)
    return mod

cdef double tnorm2(double[::1] x) nogil:
    cdef Py_ssize_t i
    cdef double s, v
    
    s = 1
    for i in range(1, <Py_ssize_t>x.shape[0]):
        v = x[i]
        s += v * v
    return s

cdef class TLinearModel(Model):
    __doc__ = """TLinearModel(param)"""
    #
    def __init__(self, o):
        if isinstance(o, (int, long)):
            self.n_input = o
            self.n_param = o + 1
            self.param = self.ob_param = None
            self.grad = None
            self.grad_input = None
        else:
            self.param = self.ob_param = np.asarray(o, 'd')
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_input = np.zeros(self.n_input, 'd')
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
        """
        y = (w_0+w_1 x_1+...+w_n x_n) / sqrt(1 + ||w||^2)
        """
        cdef Py_ssize_t i, n_input = self.n_input
        cdef double v
        cdef double[::1] param = self.param
        cdef double pn = sqrt(tnorm2(param))

        v = param[0]
        for i in range(n_input):
            v += param[i+1] * X[i]
        v -= X[n_input]
        return v / pn
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i, n_input = self.n_input
        cdef double[::1] param = self.param
        cdef double pn = sqrt(tnorm2(param))
        cdef double s

        grad[0] = 1.0 / pn
        for i in range(n_input):
            grad[i+1] = X[i] / pn

        s = param[0]
        for i in range(n_input):
            s += param[i+1] * X[i]
        s -= X[n_input]
        s /= (pn * pn * pn)

        for i in range(1, <Py_ssize_t>self.n_param):
            grad[i] -= param[i] * s
    # 
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        cdef Py_ssize_t i
        cdef double[::1] param = self.param
        cdef double pn = sqrt(tnorm2(param))

        for i in range(self.n_input):
            grad_input[i] = param[i+1] / pn
    #
    cpdef Model copy(self, bint share=1):
        cdef TLinearModel mod = TLinearModel(self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param, 'd')
        mod.grad_input = np.zeros(self.n_input, 'd')
        return <Model>mod
    #
    def as_linearmodel(self):
        cdef double[::1] param = np.array(self.param, copy=True)        
        return LinearModel(param)        
    #
    # def _repr_latex_(self):
    #     if self.param[0]:
    #         text = format_double % self.param[0]
    #     else:
    #         text = ''
    #     m = self.n_param
    #     for i in range(1, m):
    #         par = self.param[i]
    #         if fabs(par) < display_precision:
    #             continue
    #         spar = format_double % par
    #         if self.param[i] >= 0:
    #             text += "+%sx_{%s}" % (spar, i)
    #         else:
    #             text += "%sx_{%s}" % (spar, i)
    #     text = "$y(\mathbf{x})=" + text + "$"
    #     return text
    #


cdef class SigmaNeuronModel(Model):
    #
    __doc__ = "Модель сигмоидального нейрона с простыми синапсами"
    #
    def __init__(self, Func outfunc, o):
        self.outfunc = outfunc
        if isinstance(o, (int, long)):
            self.n_param = o + 1
            self.n_input = o
            self.param = self.ob_param = None
            self.grad = None
            self.grad_input = None
        else:
            self.param = self.ob_param = o
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_input = np.zeros(self.n_input, 'd')
    #
    cpdef Model copy(self, bint share=1):
        cdef Py_ssize_t n_param = self.n_param
        cdef SigmaNeuronModel mod = SigmaNeuronModel(self.outfunc, self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = self.grad.copy()
        mod.grad_input = np.zeros(self.n_input, 'd')
        return <Model>mod
    #
    cdef double _evaluate(self, double[::1] X):
        # cdef Py_ssize_t i
        # cdef double[::1] param = self.param
        cdef double s
        cdef double *param = &self.param[0]

        s =  param[0] + inventory._conv(&X[0], &param[1], self.n_input)
        # s = param[0]
        # for i in range(self.n_input):
        #     s += param[i+1] * X[i]
        
        return self.outfunc._evaluate(s)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        cdef Py_ssize_t i
        cdef double[::1] param = self.param
        cdef double s, sx
        
        s =  param[0] + inventory._conv(&X[0], &param[1], self.n_input)
        # s = param[0]
        # for i in range(self.n_input):
        #     s += param[i+1] * X[i]

        sx = self.outfunc._derivative(s)

        grad[0] = sx
        inventory._mul_set(&grad[1], &X[0], sx, self.n_input)
        # for i in range(self.n_input):
        #     grad[i+1] = sx * X[i]
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        cdef Py_ssize_t i
        cdef Py_ssize_t n_input = self.n_input
        cdef double s, sx
        cdef double[::1] param = self.param
                                
        s =  param[0] + inventory._conv(&X[0], &param[1], self.n_input)
        # s = param[0]
        # for i in range(n_input):
        #     s += param[i+1] * X[i]

        sx = self.outfunc._derivative(s)

        inventory._mul_set(&grad_input[0], &param[1], sx, self.n_input)
        # for i in range(n_input):
        #     grad_input[i] = sx * param[i+1]
    #
    def as_dict(self):
        return { 'name': 'sigma_neuron', 
                 'func': self.outfunc.to_dict(),
                 'param': (list(self.param) if self.param is not None else None), 
                 'n_input': self.n_input }
    #
    def init_from(self, ob):
        cdef double[::1] param = np.array(ob['param'], 'd')
        inventory.move(self.param, param)

@register_model('sigma_neuron')
def sigma_neuron_from_dict(ob):
    mod = SigmaNeuronModel(func_from_dict(ob['func']), ob['n_input'])
    return mod        

cdef class SimpleComposition(Model):
    #
    def __init__(self, Func func, Model model):
        self.func = func
        self.model = model
        self.n_input = model.n_input
        self.n_param = model.n_param
        self.ob_param = model.ob_param
        self.param = model.param
        self.grad = model.grad
        self.grad_input = model.grad_input
    #
    cdef double _evaluate(self, double[::1] X):
        return self.func._evaluate(self.model._evaluate(X))
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        # cdef Py_ssize_t j
        cdef double val
        cdef Model mod = self.model

        val = self.func._derivative(mod._evaluate(X))
        mod._gradient(X, grad)
        inventory._mul_const(&grad[0], val, <Py_ssize_t>grad.shape[0])
        # for j in range(self.grad.shape[0]):
        #     grad[j] *= val
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        # cdef Py_ssize_t j
        cdef double val
        cdef Model mod = self.model

        inventory._clear(&grad_input[0], <Py_ssize_t>grad_input.shape[0])
        val = self.func._derivative(mod._evaluate(X))
        mod._gradient_input(X, grad_input)
        inventory._mul_const(&grad_input[0], val, <Py_ssize_t>grad_input.shape[0])
        # for j in range(grad_input.shape[0]):
        #     grad_input[j] *= val * mod.grad_input[j]
    #        
    cpdef copy(self, bint share=1):
        return SimpleComposition(self.func, self.model)

cdef class ModelComposition(Model):
    #
    def __init__(self, Func2 func):
        self.func = func
        self.models = []
        self.n_param = 0
        self.param = None
        self.n_input = -1
        self.ss = self.sx = None
    #
    def append(self, Model mod):
        self.models.append(mod)
        self.n_param += mod.n_param
        if self.n_input < 0:
            self.n_input = mod.n_input
        elif self.n_input > 0 and mod.n_input != self.n_input:
            raise RuntimeError(f"n_input != mod.n_input: {mod}")
    #
    def extend(self, models):
        for mod in models:
            self.append(mod)
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
    #
    def _allocate_param(self, Allocator allocator):
        suballocator = allocator.suballocator()
        for mod in self.models:
            mod._allocate_param(suballocator)

        self.ob_param = suballocator.get_allocated()

        if self.param is not None:
            with cython.boundscheck(True):
                self.ob_param[:] = self.param

        self.param = self.ob_param
        self.n_param = <Py_ssize_t>self.param.shape[0]

        self.ss = np.zeros(len(self.models), 'd')
        self.sx = np.zeros(len(self.models), 'd')
        self.grad = np.zeros(self.n_param, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    cpdef copy(self, bint share=1):
        cdef ModelComposition md = ModelComposition(self.func)
        md.models = self.models[:]
        md.n_param = self.n_param
        if share:
            md.param = self.param[:]
        else:
            md.param = self.param.copy()
        return md
    #
    cdef double _evaluate(self, double[::1] X):
        cdef double w, s
        cdef Model mod
        cdef list models = self.models
        cdef Py_ssize_t j, n_models = len(models)
        cdef double[::1] ss = self.ss

        if ss is None or <Py_ssize_t>ss.shape[0] != n_models:
            ss = self.ss = np.empty(n_models, 'd')
            
        for j in range(n_models):
            # mod = <Model>models[j]
            ss[j] = (<Model>models[j])._evaluate(X)
        
        return self.func._evaluate(ss)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        
        cdef list models = self.models
        cdef Py_ssize_t i, j, n_models = len(self.models)
        cdef Py_ssize_t k, k2
        cdef double[::1] sx = self.sx
        cdef double[::1] ss = self.ss
        cdef double[::1] mod_grad
        cdef double sx_j

        if ss is None or <Py_ssize_t>ss.shape[0] != n_models:
            ss = self.ss = np.empty(n_models, 'd')
        
        if sx is None or <Py_ssize_t>sx.shape[0] != n_models:
            sx = self.sx = np.empty(n_models, 'd')
        else:
            sx = self.sx

        for j in range(n_models):
            ss[j] = (<Model>models[j])._evaluate(X)
            
        self.func.gradient(ss, sx)

        k = 0
        for j in range(n_models):
            mod = <Model>self.models[j]
            k2 = k + mod.n_param
            mod_grad = grad[k:k2]
            mod._gradient(X, mod_grad)
            sx_j = sx[j]
            for i in range(mod.n_param):
                mod_grad[i] *= sx_j
            k = k2

    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        
        cdef list models = self.models
        cdef Py_ssize_t i, j, n_models = len(self.models)
        cdef Py_ssize_t k, k2
        cdef double[::1] sx = self.sx
        cdef double[::1] ss = self.ss
        cdef double[::1] mod_grad
        cdef double sx_j

        if ss is None or <Py_ssize_t>ss.shape[0] != n_models:
            ss = self.ss = np.empty(n_models, 'd')
        
        if sx is None or <Py_ssize_t>sx.shape[0] != n_models:
            sx = self.sx = np.empty(n_models, 'd')
        else:
            sx = self.sx

        for j in range(n_models):
            ss[j] = (<Model>models[j])._evaluate(X)
            
        self.func.gradient(ss, sx)

        inventory._clear(&grad_input[0], grad_input.ahape[0])
        for j in range(n_models):
            mod = <Model>self.models[j]
            mod._gradient_input(X, self.grad_input)
            sx_j = sx[j]
            for i in range(self.n_input):
                grad_input[i] += sx_j * self.grad_input[i]
            
    cdef void _gradient_j(self, double[::1] X, Py_ssize_t j, double[::1] grad):
        # cdef Py_ssize_t i, m = grad.shape[0]
        cdef double gval
        
        gval = self.func._gradient_j(X, j)
        (<Model>self.models[j])._gradient(X, grad)
        inventory._mul_const(&grad[0], gval, <Py_ssize_t>grad.shape[0])
        # for i in range(m):
        #     grad[i] *= gval
    #
        
cdef class ModelComposition_j(Model):
    #
    def __init__(self, model_comp, j):
        self.model_comp = model_comp
        self.model_j = model_comp.models[j]
        self.n_param = self.model_j.n_param
        self.ob_param = self.model_j.ob_param
        self.param = self.model_j.param
        self.n_input = model_comp.n_input
        self.j = j
    #
    cdef double _evaluate(self, double[::1] X):
        return self.model_j._evaluate(X)
    #
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        self.model_comp._gradient_j(X, self.j, grad)
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] grad_input):
        raise RuntimeError('not implemented')

        
cdef class Model2:

    cdef void forward(self, double[::1] X):
        pass
    #
    cdef void gradient_j(self, Py_ssize_t j, double[::1] X, double[::1] grad):
        pass
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        pass
    #
    
cdef class ModelLayer(Model2):
    
    # cdef void _forward(self):
    #     pass
    # #
    # cdef void _backward(self, double[::1] grad_out, double[::1] grad):
    #     pass
    #
    cpdef ModelLayer copy(self, bint share=1):
        pass
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
    #
    # cdef double[::1] _evaluate(self, double[::1] X):
    #     self.forward(X)
    #     return self.output
    #

cdef class ScaleLayer(ModelLayer):
    #
    def _allocate_param(self, allocator):
        pass
    #
    def init_param(self):
        pass
    #
    def __init__(self, Func func, n_input):
        self.func = func
        self.n_input
        self.obparam = param = None
        self.n_param = 0
        self.n_input = n_input
        self.n_output = n_input
    #
    cdef void forward(self, double[::1] X):
        self.func._evaluate_array(&X[0], &self.output[0], self.n_input)
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        self.func._derivative_array(&X[0], &self.grad_input[0], self.n_input)
        inventory._mul(&self.grad_input[0], &grad_out[0], self.n_input)
            

cdef class SimpleLayer(ModelLayer):
    #
    cdef void forward(self, double[::1] X):
        cdef Func func = self.func
        cdef double[::1] param = self.param
        cdef double[::1] output = self.output
        cdef Py_ssize_t i, n = self.n_input
        
        for i in range(n):
            output[i] = func._evaluate(X[i] - param[i])
    #
    def init_param(self):
        if self.param is None:
            pass
            # XXXXXXX
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Func func = self.func
        cdef double[::1] param = self.param
        cdef Py_ssize_t i, n = self.n_input
        
        for i in range(n):
            grad[i] = -func._derivative(X[i] - param[i]) * grad_out[i]

    
cdef class GeneralModelLayer(ModelLayer):
    #
    def __init__(self, n_input):
        self.n_input = n_input
        self.n_output = 0
        self.n_param = 0
        self.param = self.ob_param = None
        self.models = []
        self.grad_input = None
        self.output = None
    #
    def _allocate_param(self, allocator):
        """Allocate mod.param and mod.grad for all models"""
        layer_allocator = allocator.suballocator()
        for mod in self.models:
            if mod.n_param == 0:
                mod.param = None
                continue

            mod._allocate_param(layer_allocator)

        self.ob_param = layer_allocator.get_allocated()

        if self.param is not None:
            with cython.boundscheck(True):
                self.ob_param[:] = self.param

        self.param = self.ob_param
        self.n_param = <Py_ssize_t>self.param.shape[0]

        self.n_output = len(self.models)

        self.output = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
        # self.grad = np.zeros(self.n_param, 'd')
    #
    cpdef ModelLayer copy(self, bint share=1):
        cdef GeneralModelLayer layer = GeneralModelLayer(self.n_input)
        cdef list models = layer.models
        cdef Model mod

        for mod in self.models:
            models.append(mod.copy(share))

        layer.n_output = self.n_output
        layer.param = self.param
        layer.ob_param = self.ob_param
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

        for j in range(self.n_output):
            # mod = <Model>self.models[j]
            self.output[j] = (<Model>self.models[j])._evaluate(X)
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Model mod_j
        cdef Py_ssize_t i, j, k, n_param_j
        cdef double val_j
        # cdef Py_ssize_t n_output = self.n_output
        # cdef double[::1] grad_in = self.grad_input

        inventory.clear(self.grad_input)
        k = 0
        for j in range(self.n_output):
            val_j = grad_out[j]
            mod_j = <Model>self.models[j]
            n_param_j = mod_j.n_param
            if n_param_j > 0:
                mod_j._gradient(X, mod_j.grad)
                inventory._mul_set(&grad[k], &mod_j.grad[0], val_j, n_param_j)
                k += n_param_j
                # for i in range(n_param_j):
                #     grad[k] = mod_j.grad[i] * val_j
                #     k += 1

            mod_j._gradient_input(X, mod_j.grad_input)
            inventory._mul_add(&self.grad_input[0], &mod_j.grad_input[0], val_j, self.n_input)
            # for i in range(self.n_input):
            #     grad_in[i] += mod_j.grad_input[i] * val_j
        #
    #
    cdef void gradient_j(self, Py_ssize_t j, double[::1] X, double[::1] grad):        
        (<Model>self.models[j])._gradient(X, grad)
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
        self.param = self.ob_param = None
        self.grad_input = None
        self.output = None
        self.ss = None
        # self.first_time = 1
    #
    def _allocate_param(self, allocator):
        """Allocate matrix"""
        layer_allocator = allocator.suballocator()
        self.matrix = layer_allocator.allocate2(self.n_output, self.n_input+1)
        self.param = self.ob_param = layer_allocator.get_allocated()

        self.output = np.zeros(self.n_output, 'd')
        self.ss = np.zeros(self.n_output, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    def init_param(self):
        self.ob_param[:] = self.param = np.random.random(self.n_param)
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
        cdef Py_ssize_t i, j, k
        cdef double s
        cdef double[::1] param = self.param
        cdef double[::1] output = self.output
        cdef double[::1] ss = self.ss
        cdef Func func = self.func
        cdef bint is_func = (func is not None)
         
        k = 0
        for j in range(n_output):
            s = param[k]
            k += 1
            for i in range(n_input):
                s += param[k] * X[i]
                k += 1
            ss[j] = s

        if is_func:
            for j in range(n_output):
                output[j] = func._evaluate(ss[j])
        else:
            for j in range(n_output):
                output[j] = ss[j]
    #
    cdef void backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Py_ssize_t i, j, k
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_output = self.n_output
        cdef double val_j, s, sx
        cdef double[::1] grad_in = self.grad_input

        cdef double[::1] output = self.output
        cdef double[::1] param = self.param
        cdef double[::1] ss = self.ss
        cdef Func func = self.func
        cdef bint is_func = (func is not None)
        
        k = 0
        for j in range(n_output):
            s = param[k]
            k += 1
            for i in range(n_input):
                s += param[k] * X[i]
                k += 1
            ss[j] = s

        if is_func:
            for j in range(n_output):
                ss[j] = grad_out[j] * func._derivative(ss[j])
        else:
            for j in range(n_output):      
                ss[j] = grad_out[j]

        inventory.fill(grad_in, 0)
                
        k = 0
        for j in range(n_output):  
            grad[k] = sx = ss[j]
            k += 1
            for i in range(n_input):
                grad_in[i] += sx * param[k]
                grad[k] = sx * X[i]
                k += 1
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
        inventory.move2(self.matrix, matrix)

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
    def _allocate_param(self, allocator):
        """Allocate mod.param and mod.grad for all models"""
        
        layers_allocator = allocator.suballocator()
        for layer in self.layers:
            if layer.n_param > 0:
                layer._allocate_param(layers_allocator)

        self.param = layers_allocator.get_allocated()

        n_layer = len(self.layers)
        layer = self.layers[n_layer-1]
        self.output = layer.output
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
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
        self.param = self.ob_param = None
        self.grad = None
        self.grad_input = None
    #
    def _allocate_param(self, allocator):
        ffnm_allocator = allocator.suballocator()
        if self.head.n_param > 0:
            self.head._allocate_param(ffnm_allocator)
        
        self.body._allocate_param(ffnm_allocator)
        
        self.param = self.ob_param = ffnm_allocator.get_allocated()
        self.n_param = len(self.param)
        #print("NN", allocator)
    #
    # def allocate(self):
    #     allocator = ArrayAllocator(self.n_param)
    #     self._allocate_param(allocator)
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
        head._gradient_input(body.output, head.grad_input)
        body.backward(X, head.grad_input, grad[head.n_param:])
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

#     cdef _gradient_input(self, double[::1] X, double[::1] grad):
#         cdef double x = X[0]
#         cdef double val = 1.0
#         cdef int i, m = self.param.shape[0]
        
#         for i in range(1, m):
#             grad[i] = val * i 
#             val *= x

cdef class EllipticModel(Model):
    
    def __init__(self, n_input):
        self.n_input = n_input
        self.c_size = self.n_input
        self.S_size = (self.n_input * (self.n_input + 1)) // 2
        self.n_param = self.c_size + self.S_size
        self.c = None
        self.S = None
        self.param = None
    #
    def _allocate_param(self, allocator):
        sub_allocator = allocator.suballocator()
        self.c = sub_allocator.allocate(self.c_size)
        self.S = sub_allocator.allocate(self.S_size)

        param = sub_allocator.get_allocated()
        if self.param is not None:
            param[:] = self.param
        self.param = param
    #
    def init_param(self, param=None, bint random=1):
        cdef Py_ssize_t j, k, n_input = self.n_input

        # if self.param is None:
        #     self.allocate()
        
        if param is not None:
            print('*')
            if type(param) == tuple:
                self.c[:], self.S[:] = param[0], param[1]
                return
            else:
                self.param[:] = param
        
        print(np.asarray(self.c))
        inventory.move(self.c, np.random.random(self.c_size))
        # self.c[:] = np.random.random(self.c_size)
        print(np.asarray(self.c))

        print(np.asarray(self.S))
        inventory.move(self.S, np.zeros(self.S_size, 'd'))
        # self.S[:] = np.zeros(self.S_size, 'd')
        print(np.asarray(self.S))
        k = 0
        j = n_input
        while j > 0:
            self.S[k] = 1
            k += j
            j -= 1
        print(np.asarray(self.S))

        self.grad = np.zeros(self.n_param, 'd')
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    cdef double _evaluate(self, double[::1] X):
        cdef Py_ssize_t i, j, k, n_input = self.n_input
        cdef double[::1] S = self.S
        cdef double[::1] c = self.c
        cdef double xi, v
        cdef double s
        
        k = 0
        s = 0
        for i in range(n_input):
            xi = X[i] - c[i]
            for j in range(i, n_input):
                if i == j:
                    s += xi * S[k] * xi
                else:
                    s += xi * S[k] * (X[j] - c[j])
                k += 1
        return s

    cdef _gradient_c(self, double[::1] X, double[::1] grad_c):
        cdef Py_ssize_t i, j, k, n_input = self.n_input
        cdef double[::1] S = self.S
        cdef double[::1] c = self.c
        cdef double xi
        cdef double s

        # print(grad_c.shape[0])
        inventory.fill(grad_c, 0)
        
        k = 0
        for i in range(n_input):
            for j in range(i, n_input):
                grad_c[i] += -2 * S[k] * (X[j] - c[j])
                k += 1

    cdef _gradient_S(self, double[::1] X, double[::1] grad_S):
        cdef Py_ssize_t i, j, k, n_input = self.n_input
        cdef double[::1] S = self.S
        cdef double[::1] c = self.c
        cdef double xi

        k = 0
        for i in range(n_input):
            xi = X[i] - c[i]
            for j in range(i, n_input):
                if i == j:
                    grad_S[k] = xi * xi
                else:
                    grad_S[k] = 2 * xi * (X[j] - c[j])
                k += 1
                
    cdef void _gradient(self, double[::1] X, double[::1] grad):
        # print(grad.shape[0], self.c_size)
        self._gradient_c(X, grad[:self.c_size])
        self._gradient_S(X, grad[self.c_size:])

# cdef class MahalanobisDistanceModelViewC(MahalanobisDistanceModel):
#     #
#     def __init__(self, MahalanobisDistanceModel model):
#         self.model = model
#         self.param = model.c
#         self.n_param = model.c.shape[0]
#         self.grad = model.grad
    
                
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
        self.grad_input = np.zeros(self.n_input, 'd')
    #
    cdef double _evaluate(self, double[::1] X):
        cdef double val, s
        cdef double[:,::1] matrix = self.matrix
        cdef int i, j, n, m
        
        n = matrix.shape[0]
        m = matrix.shape[1]
        val = 0
        for j in range(n):
            s = matrix[j,0]
            for i in range(m):
                s += matrix[j,i+1] * X[i]
            val += s*s
        return val
    #
    cdef void _gradient_input(self, double[::1] X, double[::1] y):
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
#     cdef void _gradient_input(self, double[::1] X, double[::1] grad):
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

