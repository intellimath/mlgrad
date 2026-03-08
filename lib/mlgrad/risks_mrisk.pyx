
cdef class MRisk(Risk):
    #
    def __init__(self, double[:,::1] X not None, double[::1] Y not None, Model model not None,
                       Loss loss=None, Average avg=None,
                       Batch batch=None):
        self.model = model
        self.param = model.param
        self.n_param = model.n_param
        self.n_input = model.n_input

        if self.model.grad is None:
            self.model.grad = np.zeros(self.n_param, np_double)

        if self.model.grad_input is None:
            self.model.grad_input = np.zeros(model.n_input, np_double)

        if loss is None:
            self.loss = ErrorLoss(Square())
        else:
            self.loss = loss

        if avg is None:
            self.avg = ArithMean()
        else:
            self.avg = avg

        if self.model.regfunc:
            self.grad_r = np.zeros(self.n_param, np_double)

        self.grad = np.zeros(self.n_param, np_double)
        self.grad_average = np.zeros(self.n_param, np_double)

        if X.shape[1] != self.n_input:
            raise ValueError('X.shape[1] != model.n_input')

        self.X = X
        self.Y = Y
        self.n_sample = len(Y)

        if batch is None:
            self.use_batch(WholeBatch(self.n_sample))
        else:
            self.use_batch(batch)

        N = len(X)
        self.weights = np.full(N, 1./N, np_double)
        self.lval = 0
        self.first = 1
    #
    cdef double _evaluate(self):
        cdef double u
        self._evaluate_losses_batch()
        u = self.avg._evaluate(self.loss_vals)
        #
        self.lval = self.avg.u
        #
        if self.model._is_regularized():
            v = self.model._evaluate_reg()
            self.lval += v

        return self.lval
    #
    cdef void _gradient(self):
        cdef Model _model = self.model
        cdef Loss _loss = self.loss

        cdef Py_ssize_t i, j, k
        cdef double yk, lval_dy, lval, vv
        #
        cdef double[:, ::1] X = self.X
        cdef double[::1] Y = self.Y
        cdef double[::1] weights = self.weights
        cdef double[::1] grad = self.grad
        cdef double[::1] grad_average = self.grad_average

        cdef Py_ssize_t[::1] indices = self.batch.indices
        cdef double[::1] model_vals = self.model_vals

        self.avg._gradient(self.loss_vals, weights)

        inventory.clear(grad_average)

        for j in range(self.batch.size):
            k = indices[j]
            # Xk = X[k]
            #
            # yk = _model._evaluate_one(Xk)
            vv = _loss._derivative(model_vals[j], Y[k]) * weights[j]

            _model._gradient_one(X[k], grad)

            for i in range(self.n_param):
                grad_average[i] += vv * grad[i]

        if self.model._is_regularized():
            self.add_regularized_gradient()
