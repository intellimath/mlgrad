
cdef class SigmaNeuronModel(Model):
    #
    __doc__ = "Модель сигмоидального нейрона с простыми синапсами"
    #
    def __init__(self, Func outfunc, o):
        self.outfunc = outfunc
        if isinstance(o, int):
            self.n_param = o + 1
            self.n_input = o
            self.param = self.ob_param = None
            self.grad = None
            self.grad_x = None
        else:
            self.param = self.ob_param = _asarray1d(o)
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param, 'd')
            self.grad_x = np.zeros(self.n_input, 'd')
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef SigmaNeuronModel _copy(self, bint share):
        cdef Py_ssize_t n_param = self.n_param
        cdef SigmaNeuronModel mod = SigmaNeuronModel(self.outfunc, self.n_input)

        if share:
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = self.grad.copy()
        mod.grad_x = np.zeros(self.n_input, 'd')
        return mod
    #
    cdef double _evaluate_one(self, double[::1] Xk):
        cdef double s

        s =  self.param[0] + inventory._dot(&self.param[1], &Xk[0], self.n_input)
        return self.outfunc._evaluate(s)
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        cdef double s, sx

        s =  self.param[0] + inventory._dot(&self.param[1], &Xk[0], self.n_input)
        sx = self.outfunc._derivative(s)

        grad[0] = sx
        inventory._mul_set(&grad[1], &Xk[0], sx, self.n_input)
    #
    cdef void _gradient_x(self, double[::1] Xk, double[::1] grad_x):
        # cdef Py_ssize_t i
        cdef double *gg = &grad_x[0]
        cdef double *xx = &Xk[0]
        cdef double s, sx

        s =  self.param[0] + inventory._dot(&self.param[1], &Xk[0], self.n_input)
        sx = self.outfunc._derivative(s)

        inventory._mul_set(&grad_x[0], &self.param[1], sx, self.n_input)
    #
    cdef void _gradient_xw(self, double[::1] Xk, double[:,::1] Gxw):
        cdef double *xx = &Xk[0]
        cdef double s, sx

        s =  self.param[0] + inventory._dot(&self.param[0], &Xk[0], self.n_input)
        ds = self.outfunc._derivative(s)
        ds2 = self.outfunc._derivative2(s)
        #
        # FIXME
        #
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
