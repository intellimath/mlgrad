def lienar_model(n_input, intercept=True):
    if intercept:
        return LinearModel(n_input)
    else:
        return DotModel(n_input)

#
# LinearModel
#

cdef class LinearModel(Model):
    __doc__ = """LinearModel(param)"""
    #
    def __init__(self, o):
        if type(o) == type(1):
            self.n_input = o
            self.n_param = o + 1
            self.param = None
            self.ob_param = None
            self.grad = None
            self.grad_base = None
            self.grad_x = None
            # self.is_allocated = 0
        else:
            self.param = self.ob_param = _asarray1d(o)
            self.param_base = self.param
            self.n_param = len(self.param)
            self.n_input = self.n_param - 1
            self.grad = np.zeros(self.n_param)
            self.grad_base = self.grad
            self.grad_x = np.zeros(self.n_input)
            # self.is_allocated = 1
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
        self.taus = None
    #
    cdef double _evaluate_one(self, double[::1] Xk):
        return self.param[0] + inventory._dot(&self.param[1], &Xk[0], self.n_input)
    #
    cdef void _evaluate(self, double[:, ::1] X, double[::1] Y):
        cdef double p0 = self.param[0]
        cdef double *param
        cdef Py_ssize_t n_input = self.n_input
        cdef double *YY = &Y[0]

        param = &self.param[1]
        for k in range(X.shape[0]):
            YY[k] = p0 + inventory._dot(param, &X[k,0], n_input)
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        grad[0] = 1.
        inventory._move(&grad[1], &Xk[0], self.n_input)
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        cdef double *param = &self.param[1]

        inventory._move(&grad_x[0], param, self.n_input)
    #
    cdef void _gradient_xw(self, double[::1] X, double[:,::1] Gxw):
        # cdef Py_ssize_t n_param = self.n_param
        cdef Py_ssize_t n_input = self.n_input
        cdef Py_ssize_t i

        inventory._fill(&Gxw[0,0], 0, n_input*n_input)
        for i in range(n_input):
            Gxw[i,i] = 1
    #
    def copy(self, bint share):
        cdef LinearModel mod = LinearModel(self.n_input, self.has_intercept)

        if share:
            mod.ob_param = self.ob_param
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param)
        mod.grad_x = np.zeros(self.n_input)
        return mod
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
    #
    def copy(self):
        mod = LinearModel(self.param)
        return mod

@register_model('linear')
def linear_model_from_dict(ob):
    mod = LinearModel(ob['n_input'])
    mod.allocate()
    param = ob.get('param', None)
    mod.init_param(np.asarray(param), random=0)
    return mod

#
# DotModel
#

cdef class DotModel(Model):
    __doc__ = """DotModel(param)"""
    #
    def __init__(self, o):
        if type(o) == type(1):
            self.n_input = o
            self.n_param = o
            self.param = self.ob_param = None
            self.param_base = None
            self.grad = None
            self.grad_x = None
            self.param = self.ob_param = None
            # self.is_allocated = 0
        else:
            self.param = self.ob_param = _asarray1d(o)
            self.param_base = self.param
            self.n_param = len(self.param)
            self.n_input = self.n_param
            self.grad = np.zeros(self.n_param)
            self.grad_x = np.zeros(self.n_input)
            # self.is_allocated = 1
        self.mask = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    cdef double _evaluate_one(self, double[::1] Xk):
        return inventory._dot(&self.param[0], &Xk[0], self.n_input)
    #
    cdef void _evaluate(self, double[:, ::1] X, double[::1] Y):
        cdef double *param = &self.param[0]
        cdef Py_ssize_t n_input = self.n_input
        cdef double *YY = &Y[0]

        for k in range(X.shape[0]):
            YY[k] = inventory._dot(param, &X[k,0], n_input)
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        inventory._move(&grad[0], &Xk[0], self.n_param)
    #
    cdef void _gradient_x(self, double[::1] X, double[::1] grad_x):
        inventory._move(&grad_x[0], &self.param[0], self.n_param)
    #
    def copy(self, bint share):
        cdef DotModel mod = DotModel(self.n_input)

        if share:
            mod.ob_param = self.ob_param
            mod.param = self.param
        else:
            mod.param = self.param.copy()

        mod.grad = np.zeros(self.n_param)
        mod.grad_x = np.zeros(self.n_input)
        return mod
    #
    def _repr_latex_(self):
        text = ''
        m = self.n_param
        for i in range(m):
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
        return { 'name': 'dot',
                 'param': (np.asarray(self.param) if self.param is not None else None),
                 'n_input': self.n_input }
    #
    def init_from(self, ob):
        cdef double[::1] param = np.array(ob['param'], 'd')
        if self.param is None:
            self.param = param
        else:
            self.param[:] = param
        self.n_param = <Py_ssize_t>param.shape[0]
        self.n_input = self.n_param
    #
    def copy(self):
        mod = DotModel(self.param)
        return mod

@register_model('dot')
def dot_model_from_dict(ob):
    mod = DotModel(ob['n_input'])
    mod.allocate()
    param = ob.get('param', None)
    mod.init_param(np.asarray(param), random=0)
    return mod

cdef class LinearModel_Normalized2(Model):
    #
    def init_param(self, param=None, random=1):
        Model.init_param(self, param=param, random=random)
        self.normalize()
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        cdef double val

        LinearModel._gradient_one(self, Xk, grad)
        val = inventory._dot(&Xk[0], &self.param[1], self.n_input)
        inventory._imul_add(&grad[1], &self.param[1], -val, self.n_input)
    #
    cpdef normalize(self):
        cdef double normval2 = inventory._dot(&self.param[1], &self.param[1], self.n_input)
        inventory._imul_const(&self.param[0], 1/sqrt(normval2), self.n_param)

#
# LinearFuncModel
#

cdef class LinearFuncModel(Model):
    #
    def __init__(self):
        self.models = []
        self.weights = list_double(0)
        self.mask = None
        self.n_input = 0
        self.param = None
        self.n_param = 0
        self.grad = None
        self.grad_x = None
        #
        self.regfunc = None
        self.tau = 0
        self.eqns = None
    #
    def add(self, Model mod, weight=1.0):
        self.models.append(mod)
        self.weights.append(weight)
        if self.n_input == 0:
            self.n_input = mod.n_input
        if self.n_input != mod.n_input:
            raise TypeError(f"n_input != mod.n_input")
    #
    def extend(self, models, weights):
        for mod, w in zip(models, weights):
            self.add(mod, w)
    #
    def copy(self, bint share):
        cdef LinearFuncModel mod = LinearFuncModel()
        mod.models = self.models[:]
        mod.weights = self.weights.copy()
        return mod
    #
    def __len__(self):
        return len(self.models)
    #
    cdef double _evaluate_one(self, double[::1] X):
        cdef double s
        cdef Model mod
        cdef list models = self.models
        cdef Py_ssize_t j, m=self.weights.size
        cdef double *weights = self.weights.data

        s = 0
        for j in range(m):
            mod = <Model>models[j]
            # w = weights[j]
            s += weights[j] * mod._evaluate_one(X)
        return s
    #
    def evaluate(self, double[:,::1] X):
        cdef Py_ssize_t k, N = len(X)
        cdef double[::1] YY

        Y = inventory.empty_array(N)
        YY = Y
        for k in range(N):
            YY[k] = self._evaluate_one(X[k])
        return Y
    #
    cdef void _gradient_one(self, double[::1] Xk, double[::1] grad):
        pass
    #
    def copy(self):
        mod = LinearFuncModel()
        for i in range(len(self)):
            mod.add(self.models[i], self.weights[i])
        return mod
    #
    def as_dict(self):
        d = {}
        d['name'] = 'linear_func'
        d['models'] = [mod.as_dict() for mod in self.models]
        d['weights'] = list(self.weights)
        return d
    #

@register_model('linear_func')
def linear_func_model_from_dict(ob):
    lfm = LinearFuncModel()
    models = [model_from_dict(d) for d in ob['models']]
    weights = ob['weights']
    for mod, w in zip(models, weights):
        lfm.add(mod, w)
    return lfm

