cdef class LinearLayer(ModelLayer):

    def __init__(self, n_input, n_output):
        self.n_input = n_input
        self.n_output = n_output
        self.n_param = (n_input+1)*n_output
        self.matrix = None
        self.param = self.ob_param = None
        self.grad_x = None
        self.output = None
        # self.first_time = 1
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    def _allocate_param(self, allocator):
        """Allocate matrix"""
        layer_allocator = allocator.suballocator()
        self.matrix = layer_allocator.allocate2(self.n_output, self.n_input+1)
        self.ob_param = layer_allocator.get_allocated()
        self.param = self.ob_param
        self.param_base = layer_allocator.buf_array
        layer_allocator.close()

        self.output = np.zeros(self.n_output, 'd')
        self.grad_x = np.zeros(self.n_input, 'd')
    #
    def init_param(self, random=True):
        if random:
            ob_param = np.ascontiguousarray(np.random.random(self.n_param)-0.5)
        else:
            ob_param = np.zeros(self.n_param, "d")
        if self.param is None:
            self.ob_param = ob_param
            self.param = self.ob_param
            self.param_base = self.param
        else:
            if len(ob_param) != self.n_param:
                raise TypeError(f"len(param) = {len(ob_param)} n_param = {self.n_param}")
            inventory.move(self.param, ob_param)
    #
    # cdef LinearLayer _copy(self, bint share):
    #     cdef LinearLayer layer = LinearLayer(self.n_input, self.n_output)

    #     layer.matrix = self.matrix
    #     layer.param = self.param

    #     layer.output = np.zeros(self.n_output, 'd')
    #     layer.grad_x = np.zeros(self.n_input, 'd')
    #     return layer
    #
    cdef void _forward(self, double[::1] X):
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t j, offset
        cdef double[::1] output = self.output

        for j in range(self.n_output):
            offset = j * (n_input+1)
            output[j] = inventory._linear_func(&self.param[offset], &X[0], n_input)
    #
    cdef void _backward(self, double[::1] Xk, double[::1] grad_out, double[::1] grad):
        cdef Py_ssize_t i, j, offset
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t n_output = self.n_output

        for j in range(n_output):
            offset = j * (n_input + 1)
            inventory._mul_set1(&grad[offset], &Xk[0], grad_out[j], n_input)

        for i in range(n_input):
            self.grad_x[i] = inventory._dot_t(&grad_out[0], &self.param[i+1], n_output, n_input+1)
    #
    cdef bint _is_regularized(self) noexcept nogil:
        if self.regfunc is None or self.tau == 0:
            return 0
        else:
            return 1
    # 
    cdef double _evaluate_reg(self):
        cdef Py_ssize_t i, j
        cdef double s = 0

        for i in range(self.n_output):
            s += self.tau * self.regfunc._evaluate(self.matrix[i])
        return s
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        cdef Py_ssize_t i, offset
        cdef Py_ssize_t n_input1 = self.n_input+1
        cdef double[::1] _grad_reg = inventory.empty_array(n_input1)

        offset = 0
        for i in range(self.n_output):
            self.regfunc._gradient(self.matrix[i], _grad_reg)
            inventory._imul_const(&_grad_reg[0], self.tau, n_input1)
            inventory._move(&grad_reg[offset], &_grad_reg[0], n_input1)
            offset += n_input1
    #
    def as_dict(self):
        return { 'name': 'linear_layer',
                 'n_input': self.n_input,
                 'n_output': self.n_output,
                 'matrix': [list(row) for row in self.matrix]
               }
    #
    def init_from(self, ob):
        cdef double[:,::1] matrix = np.array(ob['matrix'], 'd')
        inventory.move2(self.matrix, matrix)

@cython.final
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
        self.ob_param = param = None
        self.n_param = 0
        self.n_input = n_input
        self.n_output = n_input
        self.output = np.zeros(n_input, 'd')
        self.grad_x = np.zeros(n_input, 'd')
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef void _forward(self, double[::1] X):
        cdef double[::1] output = self.output
        cdef Func func = self.func
        cdef Py_ssize_t j

        for j in range(self.n_output):
            output[j] = func._evaluate(X[j])
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef double[::1] grad_x = self.grad_x
        cdef Func func = self.func
        cdef Py_ssize_t j

        for j in range(self.n_input):
            grad_x[j] = grad_out[j] * func._derivative(X[j])
    #
    def copy(self, bint share):
        cdef ScaleLayer layer = ScaleLayer(self.func, self.n_input)

        layer.param = self.param

        layer.output = np.zeros(self.n_output, 'd')
        layer.grad_x = np.zeros(self.n_input, 'd')

        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None

        return layer
    #
    cdef double _evaluate_reg(self):
        return 0
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        pass
    #
    cdef bint _is_regularized(self) noexcept nogil:
        return 0

@cython.final
cdef class Scale2Layer(ModelLayer):
    #
    def _allocate_param(self, allocator):
        layer_allocator = allocator.suballocator()
        self.ob_param = layer_allocator.get_allocated()
        self.param = self.ob_param
        self.param_base = layer_allocator.buf_array
        layer_allocator.close()
    #
    def init_param(self):
        ob_param = np.ascontiguousarray(1*np.random.random(self.n_param)-0.5)
        if self.param is None:
            self.ob_param = ob_param
            self.param = self.ob_param
        else:
            inventory.move(self.param, ob_param)
            # self.param[:] = ob_param[:]
    #
    def __init__(self, ParameterizedFunc func, n_input):
        self.func = func
        self.ob_param = param = None
        self.n_input = n_input
        self.n_param = n_input
        self.n_output = n_input
        self.output = np.zeros(n_input, 'd')
        self.grad_x = np.zeros(n_input, 'd')
        self.mask = None
    #
    cdef void _forward(self, double[::1] X):
        cdef double[::1] output = self.output
        cdef ParameterizedFunc func = self.func
        cdef Py_ssize_t j
        cdef double[::1] param = self.param
        # cdef int num_threads = inventory.get_num_threads_ex(self.n_output)

        # for j in prange(self.n_output, nogil=True, schedule='static', num_threads=num_threads):
        for j in range(self.n_output):
            output[j] = func._evaluate(X[j], param[j])
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef double *grad_in = &self.grad_x[0]
        cdef ParameterizedFunc func = self.func
        cdef Py_ssize_t j
        cdef double[::1] param = self.param
        # cdef int num_threads = inventory.get_num_threads_ex(self.n_output)

        for j in range(self.n_input):
            grad[j] = grad_out[j] * func._derivative_u(X[j], param[j])

        # for j in prange(self.n_input, nogil=True, schedule='static', num_threads=num_threads):
        for j in range(self.n_input):
            grad_in[j] = grad_out[j] * func._derivative(X[j], param[j])
    #
    # cdef ScaleLayer _copy(self, bint share):
    #     cdef ScaleLayer layer = ScaleLayer(self.func, self.n_input)

    #     layer.param = self.param

    #     layer.output = np.zeros(self.n_output, 'd')
    #     layer.grad_x = np.zeros(self.n_input, 'd')
    #     return layer

cdef class GeneralModelLayer(ModelLayer):
    #
    def __init__(self, n_input):
        self.n_input = n_input
        self.n_output = 0
        self.n_param = 0
        self.mod_n_param = 0
        self.param = self.ob_param = None
        self.models = []
        self.grad_x = None
        self.output = None
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef double _evaluate_reg(self):
        cdef Py_ssize_t i, j
        cdef Model mod
        cdef double s = 0
        cdef Func2 regfunc = self.regfunc
        cdef double tau = self.tau
        cdef list models = self.models

        if self.regfunc is None or tau == 0:
            return 0

        for i in range(self.n_output):
            mod = <Model>models[i]
            s += tau * regfunc._evaluate(mod.param)
        return s
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        cdef Py_ssize_t i, k
        cdef Py_ssize_t n_p = self.mod_n_param
        cdef Model mod
        cdef double[::1] _grad_reg
        cdef Func2 regfunc = self.regfunc
        cdef double tau = self.tau
        cdef list models = self.models

        if self.regfunc is None or tau == 0:
            return

        _grad_reg = inventory.empty_array(n_p)

        inventory.clear(grad_reg)
        k = 0
        for i in range(self.n_output):
            mod = <Model>models[i]
            regfunc._gradient(mod.param, _grad_reg)
            inventory.imul_const(_grad_reg, tau)
            inventory.move(grad_reg[k:k+n_p], _grad_reg)
            k += n_p
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
        layer_allocator.close()

        if self.param is not None:
            with cython.boundscheck(True):
                self.ob_param[:] = self.param

        self.param = self.ob_param
        self.n_param = <Py_ssize_t>self.param.shape[0]

        self.n_output = len(self.models)

        self.output = np.zeros(self.n_output, 'd')
        self.grad_x = np.zeros(self.n_input, 'd')
    #
    def init_param(self):
        for mod in self.models:
            mod.init_param()
        # self.grad = np.zeros(self.n_param, 'd')
    #
    def copy(self, bint share=0):
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
        layer.grad_x = np.zeros((self.n_input,), 'd')
        return layer
    #
    def add(self, Model mod):
        if self.n_input != mod.n_input:
            raise ValueError("layer.n_input: %s != model.n_input: %s" % (self.n_input, mod.n_input))
        if self.mod_n_param == 0:
            self.mod_n_param = mod.n_param
        else:
            if mod.n_param != self.mod_n_param:
                raise ValueError("models have different n_param")
        self.models.append(mod)
        self.n_param += self.mod_n_param
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
    cdef void _forward(self, double[::1] X):
        cdef Model mod
        cdef Py_ssize_t j
        cdef double[::1] output = self.output

        for j in range(self.n_output):
            mod = <Model>self.models[j]
            output[j] = mod._evaluate_one(X)
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef Model mod_j
        cdef Py_ssize_t i, j, k, n_param_j
        cdef double val_j
        cdef double[::1] grad_j
        # cdef Py_ssize_t n_output = self.n_output
        # cdef double[::1] grad_in = self.grad_x

        inventory.clear(self.grad_x)
        k = 0
        for j in range(self.n_output):
            val_j = grad_out[j]
            mod_j = <Model>self.models[j]
            n_param_j = mod_j.n_param
            if n_param_j > 0:
                grad_j = grad[k:k+n_param_j]
                mod_j._gradient_one(X, grad_j)
                inventory._imul_const(&grad_j[0], val_j, n_param_j)
                k += n_param_j
                # for i in range(n_param_j):
                #     grad[k] = mod_j.grad[i] * val_j
                #     k += 1

            mod_j._gradient_x(X, mod_j.grad_x)
            inventory._imul_add(&self.grad_x[0], &mod_j.grad_x[0], val_j, self.n_input)
            # for i in range(self.n_input):
            #     grad_in[i] += mod_j.grad_x[i] * val_j
        #
    #
    # cdef void gradient_j(self, Py_ssize_t j, double[::1] X, double[::1] grad):
    #     (<Model>self.models[j])._gradient_one(X, grad)
    # #
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

# cdef class SigmaNeuronModelLayer(ModelLayer):

#     def __init__(self, Func func, n_input, n_output):
#         self.n_input = n_input
#         self.n_output = n_output
#         self.n_param = (n_input+1)*n_output
#         self.func = func
#         self.matrix = None
#         self.param = self.ob_param = None
#         self.grad_x = None
#         self.output = None
#         self.ss = None
#         # self.first_time = 1
#     #
#     def _allocate_param(self, allocator):
#         """Allocate matrix"""
#         layer_allocator = allocator.suballocator()
#         self.matrix = layer_allocator.allocate2(self.n_output, self.n_input+1)
#         self.param = self.ob_param = layer_allocator.get_allocated()
#         layer_allocator.close() 

#         self.output = np.zeros(self.n_output, 'd')
#         self.ss = np.zeros(self.n_output, 'd')
#         self.grad_x = np.zeros(self.n_input, 'd')
#     #
#     def init_param(self):
#         self.ob_param[:] = self.param = np.random.random(self.n_param)
#     #
#     cpdef ModelLayer copy(self, bint share=1):
#         cdef SigmaNeuronModelLayer layer = SigmaNeuronModelLayer(self.func, self.n_input, self.n_output)
#         cdef list models = self.models
#         cdef Model mod

#         layer.matrix = self.matrix
#         layer.param = self.param

#         layer.output = np.zeros(self.n_output, 'd')
#         layer.ss = np.zeros(self.n_output, 'd')
#         layer.grad_x = np.zeros(self.n_input, 'd')
#         return <ModelLayer>layer
#     #
#     cdef void _forward(self, double[::1] X):
#         cdef Py_ssize_t n_input = self.n_input
#         cdef Py_ssize_t n_output = self.n_output
#         cdef Py_ssize_t i, j, k
#         cdef double s
#         cdef double[::1] param = self.param
#         cdef double[::1] output = self.output
#         cdef double[::1] ss = self.ss
#         cdef Func func = self.func
#         cdef bint is_func = (func is not None)
         
#         k = 0
#         for j in range(n_output):
#             s = param[k]
#             k += 1
#             for i in range(n_input):
#                 s += param[k] * X[i]
#                 k += 1
#             ss[j] = s

#         if is_func:
#             for j in range(n_output):
#                 output[j] = func._evaluate_one(ss[j])
#         else:
#             for j in range(n_output):
#                 output[j] = ss[j]
#     #
#     cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
#         cdef Py_ssize_t i, j, k
#         cdef Py_ssize_t n_input = self.n_input
#         cdef Py_ssize_t n_output = self.n_output
#         cdef double val_j, s, sx
#         cdef double[::1] grad_in = self.grad_x

#         cdef double[::1] output = self.output
#         cdef double[::1] param = self.param
#         cdef double[::1] ss = self.ss
#         cdef Func func = self.func
#         cdef bint is_func = (func is not None)
        
#         k = 0
#         for j in range(n_output):
#             s = param[k]
#             k += 1
#             for i in range(n_input):
#                 s += param[k] * X[i]
#                 k += 1
#             ss[j] = s

#         if is_func:
#             for j in range(n_output):
#                 ss[j] = grad_out[j] * func._derivative(ss[j])
#         else:
#             for j in range(n_output):      
#                 ss[j] = grad_out[j]

#         inventory.fill(grad_in, 0)
                
#         k = 0
#         for j in range(n_output):  
#             grad[k] = sx = ss[j]
#             k += 1
#             for i in range(n_input):
#                 grad_in[i] += sx * param[k]
#                 grad[k] = sx * X[i]
#                 k += 1
#     #
#     def as_dict(self):
#         return { 'name': 'sigma_neuron_layer',
#                  'func': self.func.to_dict(),
#                  'n_input': self.n_input, 
#                  'n_output': self.n_output,
#                  'matrix': [list(row) for row in self.matrix]
#                }
#     #
#     def init_from(self, ob):
#         cdef double[:,::1] matrix = np.array(ob['matrix'], 'd')
#         inventory.move2(self.matrix, matrix)

# @register_model('sigma_neuron_layer')
# def sigma_neuron_layer_from_dict(ob):
#     layer = SigmaNeuronModelLayer(ob['n_input'], ob['n_output'])
#     return layer

@cython.final
cdef class SoftNormalizerLayer(ModelLayer):
    #
    def _allocate_param(self, allocator):
        pass
    #
    def init_param(self):
        pass
    #
    def __init__(self, n_input, scale=1.0):
        self.scale = scale
        self.ob_param = param = None
        self.n_param = 0
        self.n_input = n_input
        self.n_output = n_input
        self.output = np.zeros(n_input, 'd')
        self.grad_x = np.zeros(n_input, 'd')
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef void _forward(self, double[::1] X):
        cdef double[::1] output = self.output
        cdef Func func = self.func
        cdef Py_ssize_t j
        cdef double x_max = inventory._max(&X[0], X.shape[0])
        cdef double v, s = 0
        cdef double scale = self.scale

        for j in range(self.n_output):
            output[j] = v = exp(scale*(X[j] - x_max))
            s += v
        for j in range(self.n_output):
            output[j] /= s
    #
    cdef void _backward(self, double[::1] X, double[::1] grad_out, double[::1] grad):
        cdef double[::1] grad_x = self.grad_x
        cdef double[::1] output = self.output
        cdef Py_ssize_t j, l, n = self.n_input
        cdef double x_max = inventory._max(&X[0], X.shape[0])
        cdef double v, v_j, s
        cdef double scale = self.scale

        inventory.clear(grad_x)
        for j in range(n):
            v_j = output[j]
            grad_x[j] += scale * (v_j - v_j*v_j) * grad_out[j]
            s = 0
            for l in range(n):
                if l == j:
                    continue
                v = output[l]
                s += scale * v * v_j * grad_out[l]
            grad_x[j] -= s
    #
    def copy(self, bint share):
        cdef SoftNormalizerLayer layer = SoftNormalizerLayer(self.n_input, self.scale)

        layer.param = self.param

        layer.output = np.zeros(self.n_output, 'd')
        layer.grad_x = np.zeros(self.n_input, 'd')

        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None

        return layer
    #
    cdef double _evaluate_reg(self):
        return 0
    #
    cdef void _gradient_reg(self, double[::1] grad_reg):
        pass
    #
    cdef bint _is_regularized(self) noexcept nogil:
        return 0
