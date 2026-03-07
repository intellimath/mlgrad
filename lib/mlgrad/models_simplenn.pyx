
cdef class LinearNNModel(Model):
    def __init__(self, n_input, n_hidden):
        self.linear_layer = LinearLayer(n_input, n_hidden)
        self.linear_model = LinearModel(n_hidden)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_param = (n_input+1)*n_hidden + n_hidden + 1
        self.param = self.ob_param = None
        self.grad_x = None
        self.grad = None
        # self.first_time = 1
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    def _allocate_param(self, allocator):
        _allocator = allocator.suballocator()
        self.linear_layer._allocate_param(_allocator)
        self.linear_model._allocate_param(_allocator)
        self.ob_param = _allocator.get_allocated()
        self.param = self.ob_param
        self.param_base = _allocator.buf_array
        _allocator.close()

        self.grad_x = np.zeros(self.n_input, 'd')
        self.grad = np.zeros(self.n_param, 'd')
    #
    def init_param(self):
        self.linear_layer.init_param()
        self.linear_model.init_param()
    #
    cdef double _evaluate_one(self, double[::1] Xk):
        self.linear_layer._forward(Xk)
        return self.linear_model._evaluate_one(self.linear_layer.output)
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        cdef Py_ssize_t offset = self.n_hidden*(self.n_input+1)
        cdef double[::1] grad_out = grad[offset:]

        self.linear_layer._forward(Xk)
        self.linear_model._gradient_one(self.linear_layer.output, grad_out)
        self.linear_layer._backward(Xk, grad_out, grad[:offset])
    #
    # cdef void regularizer_gradient_l2(self, double[:,::1] X, double[::1] R):

# cdef class SimpleNNModel(Model):
#     def __init__(self, n_input, n_hidden, outfunc=funcs.Sigmoidal(1.0)):
#         self.linear_layer = LinearLayer(n_input, n_hidden)
#         self.scale_layer = ScaleLayer(outfunc, n_hidden)
#         self.linear_model = LinearModel(n_hidden)
#         self.n_input = n_input
#         self.n_hidden = n_hidden
#         self.n_param = (n_input+1)*n_hidden + n_hidden + 1
#         self.param = self.ob_param = None
#         self.grad_x = None
#         self.grad = None
#         # self.first_time = 1
#         self.mask = None
#         #
#         self.regfunc = None
#         self.tau = 0
#         self.eqns = None
#         self.evaluated = 0
#     #
#     def _allocate_param(self, allocator):
#         _allocator = allocator.suballocator()
#         self.linear_layer._allocate_param(_allocator)
#         self.linear_model._allocate_param(_allocator)
#         self.ob_param = _allocator.get_allocated()
#         self.param = self.ob_param
#         self.param_base = _allocator.buf_array
#         _allocator.close()

#         self.grad_x = np.zeros(self.n_input, 'd')
#         self.grad = np.zeros(self.n_param, 'd')
#     #
#     def init_param(self):
#         self.linear_layer.init_param()
#         self.linear_model.init_param()
#     #
#     cdef double _evaluate_one(self, double[::1] Xk):
#         self.linear_layer._forward(Xk)
#         self.scale_layer._forward(self.linear_layer.output)
#         self.evaluated = 1
#         return self.linear_model._evaluate_one(self.scale_layer.output)
#     #
#     cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
#         cdef Py_ssize_t offset = self.linear_layer.n_param
#         cdef double[::1] grad_out = grad[offset:]

#         if not self.evaluated:
#             self.linear_layer._forward(Xk)
#             self.scale_layer._forward(self.linear_layer.output)

#         self.linear_model._gradient_x(self.scale_layer.output, grad_out)
#         self.scale_layer._backward(self.linear_layer.output, grad_out, None)
#         self.linear_layer._backward(self.scale_layer.output, grad_out, grad[:offset])
#         self.evaluated = 0
#     #
#     cdef void _gradient_x_one(self, double[::1] Xk, double[::1] grad):
#         cdef Py_ssize_t offset = self.linear_layer.n_input
#         cdef double[::1] grad_out = grad[offset:]

#         if not self.evaluated:
#             self.linear_layer._forward(Xk)
#             self.scale_layer._forward(self.linear_layer.output)

#         self.linear_model._gradient(self.scale_layer.output, grad_out)
#         self.scale_layer._backward(self.linear_layer.output, grad_out, None)
