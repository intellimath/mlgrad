coding: utf-8

# The MIT License (MIT)
#
# Copyright (c) <2015-2025> <Shibzukhov Zaur, szport at gmail dot com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated do cumentation files (the "Software"), to deal
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

cimport numpy
numpy.import_array()

from mlgrad.funcs import func_from_dict
import mlgrad.funcs as funcs

from cython.parallel cimport parallel, prange

cimport mlgrad.inventory as inventory

cdef int num_threads = inventory.get_num_threads()

format_double = r"%.2f"
display_precision = 0.005
np.set_printoptions(precision=3, floatmode='maxprec_equal')

cdef inline double ident(x):
    return x

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

cdef class Regularized:
    #
    def use_regularizer(self, Func2 regfunc, double tau):
        self.regfunc = regfunc
        self.tau = tau
    #
    cdef bint _is_regularized(self) noexcept nogil:
        return self.regfunc is not None and self.tau != 0
    #
    cdef double _evaluate_reg(self):
        if self.regfunc is not None or self.tau != 0:
            return self.tau * self.regfunc._evaluate(self.param)
        else:
            return 0
    #
    cdef void _gradient_reg(self, double[::1] reg_grad):
        if self.regfunc is not None or self.tau != 0:
            self.regfunc._gradient(self.param, reg_grad)
            inventory.imul_const(reg_grad, self.tau)
    #
    def evaluate_reg(self):
        return self._evaluate_reg()
    #
    def gradient_reg(self):
        reg_grad = inventory.empty_array(self.n_param)
        self._gradient_reg(reg_grad)
        return reg_grad
    #
    def use_eqn(self, Func2 eqn, tau=0):
        if self.eqns is None:
            self.eqns = []
            self.taus = list_double()
        self.eqns.append(eqn)
        self.taus.append(tau)
    #
    def use_projection(self, flag=0):
        self.projection = flag
    #
    cdef bint _with_eqns(self) noexcept nogil:
        return self.eqns is not None
    #

cdef class BaseModel(Regularized):
    #
    def allocate(self):
        """Распределить память под параметры модели.
        """
        pass
    #
    cdef double _evaluate_one(self, double[::1] x):
        return 0
    #
    def evaluate(self, X):
        """Вычисляет значения модели для всех заданных входов.
        Параметры:
            X: 2-мерный массив входов
        """
        X = _asarray2d(X)
        Y = inventory.empty_array(X.shape[0])
        self._evaluate(X, Y)
        return Y
    #
    def predict(self, X):
        return self.evaluate(X)
    #
    # def evaluate_all(self, X):
    #     X = _asarray2d(X)
    #     Y = inventory.empty_array(X.shape[0])
    #     self._evaluate(X, Y)
    #     return Y
    # #
    def evaluate_one(self, x):
        """Вычисляет значения модели для одного заданного входа.
        Параметры:
            X: 1-мерный вход
        """
        return self._evaluate_one(_asarray1d(x))
    #
    cdef void _evaluate(self, double[:,::1] X, double[::1] Y):
        cdef Py_ssize_t k, N = X.shape[0]

        for k in range(N):
            Y[k] = self._evaluate_one(X[k])
    #
    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
        pass
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        pass
    #
    cdef void _gradient_all(self, double[:,::1] X, double[:,::1] G):
        cdef Py_ssize_t k, N = X.shape[0]

        for k in range(N):
            self._gradient_one(X[k], G[k])
    #
    cdef void _gradient_x_all(self, double[:,::1] X, double[:,::1] G):
        cdef Py_ssize_t k, N = X.shape[0]

        for k in range(N):
            self._gradient_x(X[k], G[k])
    #
    def copy(self, bint share=0):
        return self._copy(share)
    #

cdef class Model(BaseModel):
    #
    def allocate(self):
        if self.param is not None:
            raise RuntimeError('param is allocated already ')
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
        allocator = ArrayAllocator(self.n_param)
        self._allocate_grad(allocator)
    #
    def _allocate_param(self, allocator):
        """
        Распределение памяти под `self.param` (если `self.param` уже создан, то его содержимое 
        переносится во вновь распределенное пространство)
        """
        self.ob_param = allocator.allocate(self.n_param)
        if not self.ob_param.flags['C_CONTIGUOUS']:
            raise TypeError("Array is not contiguous")
        if self.param is not None:
            if self.param.size != self.ob_param.size:
                raise TypeError("ob_param.size != param.size")
            with cython.boundscheck(True):
                self.ob_param[:] = self.param
        self.param = self.ob_param
        self.param_base = allocator.buf_array
        self.grad_x = np.zeros(self.n_input, 'd')
        self.mask = None
    #
    def _allocate_grad(self, allocator):
        """
        Распределение памяти под `self.grad`
        """
        self.grad = allocator.allocate(self.n_param)
        self.grad_base = self.grad
        # np.asarray(self.grad, copy=False)
    #
    def init_param(self, param=None, random=1):
        if param is None:
            if random:
                r = 0.1*np.random.random(self.n_param)-0.05
            else:
                r = np.zeros(self.n_param, 'd')
        else:
            if len(param) != self.n_param:
                raise TypeError(f"len(param) = {len(param)} n_param = {self.n_param}")
            r = param

        if self.param is None:
            self.ob_param = self.param = r
            self.param_base = self.param
        else:
            self.ob_param[:] = r
    #
    def gradient_one(self, Xk):
        grad = np.empty(self.n_param, 'd')
        self._gradient_one(Xk, grad)
        return grad
    #
    def gradient_x(self, X):
        grad_x = np.empty(self.n_input, 'd')
        self._gradient_x(X, grad_x)
        return grad_x
    #
    cdef _update_param(self, double[::1] param):
        inventory._move(&self.param[0], &param[0], self.n_param)
    #
    def update_param(self, param):
        self._update_param(param)
        # self.param[:] = param
    #
    cdef void _gradient_xw(self, double[::1] X, double[:,::1] Gxw):
        pass

include "models_linear.pyx"
include "models_neuron.pyx"

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
        self.grad_x = model.grad_x
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
#
    cdef double _evaluate_one(self, double[::1] X):
        return self.func._evaluate(self.model._evaluate_one(X))
    #
    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
        cdef double val
        cdef Model mod = self.model

        val = self.func._derivative(mod._evaluate_one(X))
        mod._gradient_one(X, grad)
        inventory._imul_const(&grad[0], val, <Py_ssize_t>grad.shape[0])
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        # cdef Py_ssize_t j
        cdef double val
        cdef Model mod = self.model

        # inventory._clear(&grad_x[0], <Py_ssize_t>grad_x.shape[0])
        val = self.func._derivative(mod._evaluate_one(X))
        mod._gradient_x(X, grad_x)
        inventory._imul_const(&grad_x[0], val, <Py_ssize_t>grad_x.shape[0])
        # for j in range(grad_x.shape[0]):
        #     grad_x[j] *= val * mod.grad_x[j]
    #
    # cdef SimpleComposition _copy(self, bint share):
    #     return SimpleComposition(self.func, self.model)

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
        self.regfunc = None
        self.tau = 0
        self.eqns = None
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
        suballocator.close()

        if self.param is not None:
            with cython.boundscheck(True):
                self.ob_param[:] = self.param

        self.param = self.ob_param
        self.n_param = <Py_ssize_t>self.param.shape[0]

        self.ss = np.zeros(len(self.models), 'd')
        self.sx = np.zeros(len(self.models), 'd')
        self.grad = np.zeros(self.n_param, 'd')
        self.grad_x = np.zeros(self.n_input, 'd')
    #
    # cdef ModelComposition _copy(self, bint share):
    #     cdef ModelComposition md = ModelComposition(self.func)
    #     md.models = self.models[:]
    #     md.n_param = self.n_param
    #     if share:
    #         md.param = self.param[:]
    #     else:
    #         md.param = self.param.copy()
    #     return md
    #
    cdef double _evaluate_one(self, double[::1] X):
        cdef double w, s
        cdef Model mod
        cdef list models = self.models
        cdef Py_ssize_t j, n_models = len(models)
        cdef double[::1] ss = self.ss

        if ss is None or <Py_ssize_t>ss.shape[0] != n_models:
            ss = self.ss = np.empty(n_models, 'd')

        for j in range(n_models):
            # mod = <Model>models[j]
            ss[j] = (<Model>models[j])._evaluate_one(X)

        return self.func._evaluate(ss)
    #
    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
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
            ss[j] = (<Model>models[j])._evaluate_one(X)

        self.func.gradient(ss, sx)

        k = 0
        for j in range(n_models):
            mod = <Model>self.models[j]
            k2 = k + mod.n_param
            mod_grad = grad[k:k2]
            mod._gradient_one(X, mod_grad)
            sx_j = sx[j]
            for i in range(mod.n_param):
                mod_grad[i] *= sx_j
            k = k2

    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
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
            ss[j] = (<Model>models[j])._evaluate_one(X)

        self.func.gradient(ss, sx)

        inventory._clear(&grad_x[0], grad_x.ahape[0])
        for j in range(n_models):
            mod = <Model>self.models[j]
            mod._gradient_x(X, self.grad_x)
            sx_j = sx[j]
            for i in range(self.n_input):
                grad_x[i] += sx_j * self.grad_x[i]

    cdef void _gradient_j(self, double[::1] X, Py_ssize_t j, double[::1] grad):
        # cdef Py_ssize_t i, m = grad.shape[0]
        cdef double gval

        gval = self.func._gradient_j(X, j)
        (<Model>self.models[j])._gradient_one(X, grad)
        inventory._imul_const(&grad[0], gval, <Py_ssize_t>grad.shape[0])
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
        self.mask = None
    #
    cdef double _evaluate_one(self, double[::1] X):
        return self.model_j._evaluate_one(X)
    #
    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
        self.model_comp._gradient_j(X, self.j, grad)
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        raise RuntimeError('not implemented')

cdef class Model2(Regularized):

    cdef void _forward(self, double[::1] Xk):
        pass
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        pass
    #
    def forward(self, Xk):
        self._forward(Xk)
    #
    def backward(self, X, grad_out, grad):
        self._backward(X, grad_out, grad)
    #
    def copy(self, bint share=0):
        return self._copy(share)
    #
    def evaluate_one(self, double[::1] Xk):
        self._forward(Xk)
        return np.array(self.output, copy=True)
    #
    def evaluate(self, X):
        n_output = self.n_output
        N = len(X)
        Y = np.empty((N, n_output), "d")
        for k, Xk in enumerate(X):
            self._forward(Xk)
            Y[k,:] = self.output
        return Y
    #

cdef class ModelLayer(Model2):
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
    #

include "models_simplenn.pyx"
include "models_layers.pyx"

cdef class MLModel:

    cdef void _forward(self, double[::1] X):
        pass
    #
    def forward(self, double[::1] Xk):
        self._forward(Xk)
    #
    def backward(self, double[::1] Xk, grad_u, double[::1] grad=None):
        if grad is None:
            self._backward(Xk, grad_u, self.grad)
        else:
            self._backward(Xk, grad_u, grad)
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        pass
    #
    cdef void backward2(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        self._forward(X)
        self._backward(X, grad_u, grad)
    #

    # def copy(self, bint share=0):
    #     return self._copy(share)
    # #
    def _allocate_param(self, allocator):
        """Allocate mod.param and mod.grad for all models"""

        layers_allocator = allocator.suballocator()
        for layer in self.layers:
            if layer.n_param > 0:
                layer._allocate_param(layers_allocator)

        self.param = layers_allocator.get_allocated()
        layers_allocator.close()

        n_layer = len(self.layers)
        layer = self.layers[n_layer-1]
        self.output = layer.output
    #
    def allocate(self):
        allocator = ArrayAllocator(self.n_param)
        self._allocate_param(allocator)
    #
    def init_param(self):
        for layer in self.layers:
            layer.init_param()
    #
    def add(self, layer):
        n_layer = len(self.layers)
        if n_layer == 0:
            self.n_input = layer.n_input
        else:
            n_output = self.layers[n_layer-1].n_output
            if n_output != layer.n_input:
                raise RuntimeError(f"Previous layer n_output={n_output}, layer n_input={layer.n_input}")
        self.layers.append(layer)

        self.n_param += layer.n_param
        self.n_output = layer.n_output
        self.output = layer.output
    #
    def __iadd__(self, layer):
        self.add(layer)
        return self
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
    # def evaluate_one(self, double[::1] Xk):
    #     self._forward(Xk)
    #     return np.array(self.output, copy=True)
    # #
    # def evaluate(self, X):
    #     n_output = self.n_output
    #     N = len(X)
    #     Y = np.empty((N, n_output), "d")
    #     for k, Xk in enumerate(X):
    #         self._forward(Xk)
    #         Y[k,:] = self.output
    #     return Y
    #
    cdef bint _is_regularized(self) noexcept nogil:
        return 1
    #
    cdef double _evaluate_reg(self):
        cdef double s = 0
        cdef Py_ssize_t j, n = len(self.layers)
        cdef ModelLayer layer

        for i in range(n):
            layer = <ModelLayer>self.layers[i]
            n_param = layer.n_param
            if n_param > 0:
                if layer._is_regularized():
                    s += layer._evaluate_reg()
        return s
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        cdef double s = 0
        cdef Py_ssize_t j, n = len(self.layers)
        cdef Py_ssize_t k
        cdef ModelLayer layer
        cdef Py_ssize_t n_param

        k = 0
        for i in range(n):
            layer = <ModelLayer>self.layers[i]
            n_param = layer.n_param
            if n_param > 0:
                if layer._is_regularized():
                    layer._gradient_reg(grad_reg[k:k+n_param])
                k += n_param


cdef class FFNetworkModel(MLModel):

    def __init__(self):
        self.n_param = 0
        self.layers = []
        self.param = None
        self.is_forward = 0
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef void _forward(self, double[::1] X):
        cdef Py_ssize_t i, n_layer
        cdef ModelLayer layer
        cdef double[::1] input, output
        cdef list layers = self.layers

        n_layer = len(self.layers)
        input = X
        for i in range(n_layer):
            layer = <ModelLayer>layers[i]
            # print(i, np.asarray(input))
            layer._forward(input)
            input = layer.output
            # print(i, np.asarray(layer.output))
        self.is_forward = 1

    cdef void _backward(self, double[::1] X, double[::1] grad_u, double[::1] grad):
        # cdef Py_ssize_t n_layer = PyList_GET_SIZE(self.layers)
        cdef Py_ssize_t n_layer = len(self.layers)
        cdef Py_ssize_t j, l, m, m0
        cdef ModelLayer layer, prev_layer
        cdef double[::1] grad_out, input
        cdef list layers = self.layers

        if not self.is_forward:
            self._forward(X)
        m = grad.shape[0]
        l = n_layer-1
        grad_out = grad_u
        while l >= 0:
            # layer = <ModelLayer>PyList_GET_ITEM(layers, l)
            layer = <ModelLayer>layers[l]
            if l > 0:
                # prev_layer = <ModelLayer>PyList_GET_ITEM(layers, l-1)
                prev_layer = <ModelLayer>layers[l-1]
                input = prev_layer.output
            else:
                input = X
            m0 = m - layer.n_param
            if layer.n_param > 0:
                layer._backward(input, grad_out, grad[m0:m])
            else:
                layer._backward(input, grad_out, None)
            grad_out = layer.grad_x
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
        self.grad_x = None
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    def _allocate_param(self, allocator):
        ffnm_allocator = allocator.suballocator()
        if self.head.n_param > 0:
            self.head._allocate_param(ffnm_allocator)

        self.body._allocate_param(ffnm_allocator)

        self.param = self.ob_param = ffnm_allocator.get_allocated()
        ffnm_allocator.close()

        self.n_param = len(self.param)
        #print("NN", allocator)
    #
    # def allocate(self):
    #     allocator = ArrayAllocator(self.n_param)
    #     self._allocate_param(allocator)
    #
    # cdef FFNetworkFuncModel _copy(self, bint share):
    #     cdef FFNetworkFuncModel mod = FFNetworkFuncModel(self.head.copy(share), self.body.copy(share))
        
    #     mod.param = self.param
        
    #     return mod
    #
    cdef double _evaluate_one(self, double[::1] X):
        self.body._forward(X)
        return self.head._evaluate_one(self.body.output)
    #
    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
        cdef int i, j, n
        cdef Model head = self.head
        cdef MLModel body = self.body
        
        # body.forward(X) 
        if head.n_param > 0:
            head._gradient_one(body.output, grad[:head.n_param])
        head._gradient_x(body.output, head.grad_x)
        body._backward(X, head.grad_x, grad[head.n_param:])
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
    cdef bint _is_regularized(self) noexcept nogil:
        return 1
    #
    cdef double _evaluate_reg(self):
        cdef double s = 0

        if self.head._is_regularized():
            s += self.head._evaluate_reg()
        if self.body._is_regularized():
            s += self.body._evaluate_reg()
        return s
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        cdef Py_ssize_t k = 0
        if self.head._is_regularized():
            self.head._gradient_reg(grad_reg[:self.head.n_param])
        if self.body._is_regularized():
            self.body._gradient_reg(grad_reg[self.head.n_param:])


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
#     cdef double _evaluate_one(self, double[::1] X):
#         cdef double *param_ptr = &self.param[0]
#         cdef double x = X[0]
#         cdef double val = 0
#         cdef int i, m = self.param.shape[0]
        
#         i = m-1
#         while i >= 0:
#             val = val * x + param_ptr[i]
#             i -= 1
#         return val
        
#     cdef _gradient_one(self, double[::1] X, double[::1] grad):
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


cdef class EuclideanNormModel(Model):
    #
    def __init__(self, double[::1] c):
        self.param = c
        self.n_input = c.shape[0]
        self.n_param = c.shape[0]
        self.mask = None
        self.regfunc = None
        self.tau = 0
        self.eqns = None
        self.grad = np.zeros(self.n_param)
        self.grad_x = np.zeros(self.n_input)
    #
    cdef double _evaluate_one(self, double[::1] x):
        cdef Py_ssize_t i, n = self.n_param
        cdef double[::1] param = self.param
        cdef double v, s = 0

        for i in range(n):
            v = param[i] - x[i]
            s += v*v
        return s/2
    #
    cdef void _gradient_one(self, double[::1] x, double[::1] g):
        cdef Py_ssize_t i, n = self.n_param
        cdef double[::1] param = self.param

        for i in range(n):
            g[i] = param[i] - x[i]
    #
    cdef void _gradient_x(self, double[::1] x, double[::1] g):
        cdef Py_ssize_t i, n = self.n_param
        cdef double[::1] param = self.param

        for i in range(n):
            g[i] = x[i] - param[i]
    #

cdef class EllipticModel(Model):
    #
    def __init__(self, n_input):
        self.n_input = n_input
        self.c_size = self.n_input
        self.S_size = (self.n_input * (self.n_input + 1)) // 2
        self.n_param = self.c_size + self.S_size
        self.c = None
        self.S = None
        self.param = None
        self.mask = None
    #
    def _allocate_param(self, allocator):
        sub_allocator = allocator.suballocator()
        self.c = sub_allocator.allocate(self.c_size)
        self.S = sub_allocator.allocate(self.S_size)

        param = sub_allocator.get_allocated()
        sub_allocator.close()

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
        self.grad_x = np.zeros(self.n_input, 'd')
    #
    cdef double _evaluate_one(self, double[::1] X):
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

    cdef void _gradient_one(self, double[::1] X, double[::1] grad):
        # print(grad.shape[0], self.c_size)
        self._gradient_c(X, grad[:self.c_size])
        self._gradient_S(X, grad[self.c_size:])



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
    cdef double _evaluate_one(self, double[::1] X):
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
    cdef void _gradient_one(self, double[::1] X, double[::1] y):
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
#     cdef double _evaluate_one(self, double[::1] X):
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
#     cdef void _gradient_one(self, double[::1] X, double[::1] grad):
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
            

# cdef class ScalarModel:
#     cdef double _evaluate_one(self, double x)
#     cdef void _gradient_one(self, double x, double[::1] grad)


cdef class LogSpecModel(Model):
    #
    def __init__(self, n_param, center=None, scale=None, param=None):
        self.n_param = n_param
        self.n_input = 1
        if param is None:
            self.param = self.ob_param = np.zeros(n_param, "d")
        else:
            self.param = self.ob_param = param
        if scale is None:
            self.scale = np.ones(n_param, "d")
        else:
            self.scale = scale
        if center is None:
            self.center = np.zeros(n_param, "d")
        else:
            self.center = center
        self.grad = np.zeros(n_param, "d")
    #
    cdef double _evaluate_one(self, double[::1] x):
        cdef double v, s
        cdef double *param = &self.param[0]
        cdef double *center = &self.center[0]
        cdef double *scale = &self.scale[0]

        cdef Py_ssize_t k
        cdef double xx = x[0]

        s = 0
        for k in range(self.n_param):
            v = (xx - center[k]) / scale[k]
            s += log(1 + param[k] * exp(-v*v/2))
        return s
    #
    cdef void _gradient_one(self, double[::1] x, double[::1] grad):
        cdef double v, ee
        cdef double *param = &self.param[0]
        cdef double *center = &self.center[0]
        cdef double *scale = &self.scale[0]

        cdef Py_ssize_t k, m = self.n_param
        cdef double xx = x[0]

        inventory._clear(&grad[0], grad.shape[0])

        for k in range(self.n_param):
            v = (xx - center[k]) / scale[k]
            ee = exp(-v*v/2)
            grad[k] =  ee / (1 + param[k] * ee)
