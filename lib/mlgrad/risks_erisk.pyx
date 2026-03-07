cdef class ERisk(Risk):
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None,
                 Loss loss=None, Batch batch=None, sample_weights=None, is_natgrad=0):

        self.model = model
        self.param = model.param
        #
        self.n_param = model.n_param
        self.n_input = model.n_input

        if loss is None:
            self.loss = SquareErrorLoss()
        else:
            self.loss = loss

        if self.model._is_regularized():
            self.grad_r = np.zeros(self.n_param, np_double)
        else:
            self.grad_r = None
        if self.model._with_eqns():
            self.grad_eqns = np.zeros(self.n_param, np_double)
        else:
            self.grad_eqns = None

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np.zeros(self.n_param, np_double)

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)

        if batch is None:
            self.use_batch(WholeBatch(self.n_sample))
        else:
            self.use_batch(batch)

        self.weights = np.full(self.n_sample, 1./self.n_sample, "d")
        if sample_weights is None:
            self.sample_weights = np.ones(self.n_sample, "d")
        else:
            self.sample_weights = np.asarray(sample_weights)
        self.lval = 0
        self.is_natgrad = is_natgrad
    #
    cdef double _evaluate(self):
        cdef Model _model = self.model
        cdef Py_ssize_t j, k, m
        cdef double S

        cdef Func2 eqn
        cdef double vl

        cdef double[::1] L = self.L
        cdef double[::1] weights = self.weights
        cdef double[::1] sample_weights = self.sample_weights
        cdef Py_ssize_t[::1] indices = self.batch.indices

        # self._evaluate_models_batch()
        self._evaluate_losses_batch()

        S = 0
        for j in range(self.batch.size):
            k = indices[j]
            S += sample_weights[k] * weights[k] * L[j]

        if _model._with_eqns():
            m = len(_model.eqns)
            for j in range(m):
                eqn = <Func2>_model.eqns[j]
                S += _model.taus[j] * eqn._evaluate(_model.param)

        if _model._is_regularized():
            S += _model._evaluate_reg()

        self.lval = S
        return S
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t j, k, m
        cdef double v
        #
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] sample_weights = self.sample_weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] model_vals = self.model_vals

        inventory.clear(grad_average)

        for j in range(self.batch.size):
            k = indices[j]

            _model._gradient_one(X[k], grad)
            v = sample_weights[k] * weights[k] * _loss._derivative(model_vals[j], Y[k])
            inventory._imul_add(&grad_average[0], &grad[0], v, self.n_param)

        if self.is_natgrad:
            s = 0
            for i in range(self.n_param):
                v = grad_average[i]
                s += v * v
            s = sqrt(s)

            for i in range(self.n_param):
                grad_average[i] /= s

        inventory.move(self.grad, self.grad_average)

        if _model._with_eqns():
            self.add_equations_gradient()

        if _model.projection:
            self.project_equations()

        if _model._is_regularized():
            self.add_regularized_gradient()

